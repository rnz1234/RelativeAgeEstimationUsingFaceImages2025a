from torch.optim.lr_scheduler import ReduceLROnPlateau


class ReduceLROnPlateauEnhanced(ReduceLROnPlateau):
	def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
		super(ReduceLROnPlateauEnhanced, self).__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)

	def set_lr(self, lr):
		self._last_lr = lr

	def get_last_lr(self):
		return self._last_lr

	def get_lr(self):
		return self.get_last_lr()