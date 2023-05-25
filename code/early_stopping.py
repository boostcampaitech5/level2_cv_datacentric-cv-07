class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=False):
        '''
        patience (int): validation loss가 증가하면 바로 종료시키지 않고, 개선을 위해 epoch을 몇 번이나 허용할 것인가.
        delta  (float): 개선되고 있다고 판단하기 위한 최소 변화량. 만약 변화량이 delta보다 적은 경우에는 개선이 없다고 판단.
        mode  (string): 관찰항목에 대해 개선이 없다고 판단하기 위한 기준을 설정.
            - min: 관찰값이 감소하는 것을 멈출 때, 학습을 종료한다. (Loss 전용)

        예시) 현재 loss가 0.612이고 best_loss가 0.621일 시 개선되고 있다고 말한다. 그러나 delta 0.01으로 설정 시 개선이 안되었다고 판단하게 된다.
        0.612 < 0.621 - 0    --> 개선 O
        0.612 < 0.621 - 0.01 --> 개선 X
        '''
        self.patience   = patience
        self.delta      = delta
        self.mode       = mode
        self.best_loss  = None
        self.early_stop = False
        self.verbose    = verbose
    
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
            self.counter = 0

        elif self.mode == 'min':
            if loss < (self.best_loss - self.delta):
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[Early Stopping] Counter: {self.counter} | Best Loss: {self.best_loss:.4f} | Current Loss: {loss:.4f}')
        
        if self.counter >= self.patience:
            self.early_stop = True
            print("[Early Stopping] Validation loss is no longer reducing. Training is finished!")
        else:
            self.early_stop = False