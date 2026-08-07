import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from models import get_param_groups
from optim import intialize_optimizer, initialize_scheduler
from data.datasets.data_prep_utils import intra_doc_causal_mask


def _move_to_device(batch, seq_len, device, intra_doc_masking):
  """Slice batch to get inputs and targets, and move them to device."""

  inputs = batch['input_ids'][:, :seq_len]
  targets = batch['input_ids'][:, 1 : (seq_len + 1)]

  if intra_doc_masking:
    # build one mask per example and stack into (bsz, L, L)
    masks = [intra_doc_causal_mask(doc_lengths, seq_len + 1, device) for doc_lengths in batch['docs_lengths']]
    attn_mask = torch.stack(masks, dim=0)  # (bsz, L+1, L+1)
    attn_mask = attn_mask[:, :seq_len, :seq_len].contiguous()  # (bsz, L, L)
  else:
    attn_mask = None

  if 'cuda' in device:
    # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.pin_memory().to(device, non_blocking=True)
  else:
    inputs, targets = inputs.to(device), targets.to(device)

  return inputs, targets, attn_mask


class TorchEngine(torch.nn.Module):
  """
  A module containing model, optimizer, scheduler, grad scaler.
  Wraps together a training step. Takes care of grad accumulation.
  """

  def __init__(self, model, cfg, device, local_rank, ckpt):
    super().__init__()

    self.micro_steps = 0
    self.accumulated_samples = 0

    self.seq_len = cfg.seq_len
    self.accumulation_steps = cfg.grad_accumulation_steps
    self.grad_clip = cfg.grad_clip
    self.dtype = cfg.dtype
    self.intra_doc_masking = getattr(cfg, 'intra_doc_masking', False)

    self.device = device

    # Whether optimizer needs loss value (e.g. Scion-NGN, NGN-MDv1, MaxMuon-Momo)
    self.pass_loss_to_optim = (
      bool(getattr(cfg, 'scion_ngn', False)) or
      cfg.optim in ('ngnmdv1', 'muonmax_momo', 'sfplus')
    )
    self._accumulated_loss = 0.0

    # Load model state dict
    if cfg.resume:
      model.load_state_dict(ckpt['state_dict'])
      self.micro_steps = ckpt['step'] * cfg.grad_accumulation_steps

    # Move model to device and to DDP
    self.model = model.to(device)
    if torch.distributed.is_initialized():
      self.model = DDP(self.model, device_ids=[local_rank])

    # Compile
    if cfg.torch_compile:
      print('Compiling the model...')
      self.model = torch.compile(self.model)

    # AMP
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
    self.scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))

    # Loss
    self.criterion = CrossEntropyLoss()

    # Optimizer
    param_groups = get_param_groups(model, cfg.weight_decay)
    self.optimizer = intialize_optimizer(param_groups, cfg, model=model)
    self.scheduler = initialize_scheduler(self.optimizer, cfg)

    # No scheduler here reads the loss; only the adaptive optimizers do, via
    # pass_loss_to_optim. Kept as a flag so the step() plumbing stays generic.
    self.scheduler_needs_loss = False

    if cfg.resume:
      self.optimizer.load_state_dict(ckpt['optimizer'])
      self.scheduler.load_state_dict(ckpt['scheduler'])
      self.scaler.load_state_dict(ckpt['scaler'])

  def step(self, batch):
    """Wraps a fwd pass, bwd pass, and optimization step."""

    self.model.train()

    self.micro_steps += 1
    self.accumulated_samples += 1

    inputs, targets, attn_mask = _move_to_device(batch, self.seq_len, self.device, self.intra_doc_masking)

    # sync (reduce) gradients at the last accumulation step
    if torch.distributed.is_initialized():
      self.model.require_backward_grad_sync = self.accumulated_samples == self.accumulation_steps

    # forward pass with autocasting
    with self.ctx:
      output = self.model(inputs, attn_mask)
      logits = getattr(output, 'logits', output)
      loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
      loss = loss / self.accumulation_steps

    # detach for logging (scale up to undo the division above)
    loss_val = loss.detach() * self.accumulation_steps
    if torch.isnan(loss_val):
      raise ValueError('Train loss is nan')

    # accumulate loss for adaptive optimizers (average over micro-batches)
    if self.pass_loss_to_optim or self.scheduler_needs_loss:
      self._accumulated_loss += loss_val.item()

    # backward pass, with gradient scaling if training in fp16
    self.scaler.scale(loss).backward()

    # step after accumulation
    if self.accumulated_samples == self.accumulation_steps:
      self.accumulated_samples = 0

      if self.grad_clip:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

      # step the optimizer, step the scaler if training in fp16
      avg_loss = None
      if self.pass_loss_to_optim or self.scheduler_needs_loss:
        avg_loss = self._accumulated_loss / self.accumulation_steps
        self._accumulated_loss = 0.0
      if self.pass_loss_to_optim:
        self.scaler.step(self.optimizer, loss=avg_loss)
      else:
        self.scaler.step(self.optimizer)
      self.scaler.update()

      # flush the gradients
      self.optimizer.zero_grad(set_to_none=True)

      # step the scheduler
      if self.scheduler:
        if self.scheduler_needs_loss:
          self.scheduler.step(loss=avg_loss)
        else:
          self.scheduler.step()

    return loss_val

  def get_optim_diagnostics(self):
    """Return optimizer diagnostics dict if available."""
    return getattr(self.optimizer, 'diagnostics', {})

  @torch.no_grad()
  def eval(self, dataloader, max_batches=100):
    """Evaluate model on a dataloader."""

    self.model.eval()
    # Schedule-free optimizers must be switched to their averaged iterate x
    # for evaluation, then back to the training point y.
    if hasattr(self.optimizer, 'eval'):
      self.optimizer.eval()

    # Compute loss on dataloader
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
      inputs, targets, attn_mask = _move_to_device(batch, self.seq_len, self.device, self.intra_doc_masking)
      with self.ctx:
        output = self.model(inputs, attn_mask)
        logits = getattr(output, 'logits', output)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

      if torch.isnan(loss) or loss is None:
        raise ValueError('Validation loss is nan')

      total_loss += loss.item()
      num_batches += 1
      if max_batches and num_batches >= max_batches:
        break

    # reduce loss across processes
    if dist.is_initialized():
      total_loss_tensor = torch.tensor([total_loss], device=self.device)
      num_batches_tensor = torch.tensor([num_batches], device=self.device, dtype=torch.int)
      dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
      total_loss = total_loss_tensor.item()
      num_batches = num_batches_tensor.item()

    # calculate average loss
    avg_loss = total_loss / num_batches

    # back to training mode (see the eval() switch at the top)
    if hasattr(self.optimizer, 'train'):
      self.optimizer.train()
    self.model.train()

    return avg_loss
