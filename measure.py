import numpy as np
class Calculator(object):
    def __init__(self,words=2):
        self.tp = np.zeros(words)
        self.tn = np.zeros(words)
        self.fp = np.zeros(words)
        self.fn = np.zeros(words)

    def record(self,ground_truth,predict):
        self.tp += np.bitwise_and(ground_truth,predict)
        self.tn += np.bitwise_and(1-ground_truth,1-predict)
        self.fp += np.bitwise_and(ground_truth,1-predict)
        self.fn += np.bitwise_and(1-ground_truth,predict)
    def mean_f1(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-8)
        return np.mean(f1)


if __name__ == "__main__":
    label_1,predict_1 = np.array([0,0,1,1,1]),np.array([0,1,1,1,1])
    label_2,predict_2 = np.array([1,0,1,1,0]),np.array([1,0,0,1,0])

    calculator = Calculator(5)
    calculator.record(label_1,predict_1)
    calculator.record(label_2,predict_2)
    print(calculator.mean_f1()) #0.7333333235555556