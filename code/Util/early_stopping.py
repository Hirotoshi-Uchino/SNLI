class EarlyStopping():
    def __init__(self, mode='loss', patience=0, verbose=0):
        self._step = 0
        self._tgt_val = float('inf')
        if mode == 'acc':
            self._tgt_val  = 0.
        self.mode  = mode
        self.patience = patience
        self.verbose = verbose


    def validate(self, tgt_val):
        if self.mode =='acc':
            tgt_val = -1 * tgt_val

        if self._tgt_val < tgt_val:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._tgt_val = tgt_val

        return False