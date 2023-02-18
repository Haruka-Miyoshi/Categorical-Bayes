import numpy as np

# カテゴリ分布の学習と予測
class Categorical:
    def __init__(self, alpha=1.0, K=6):
        # ハイパーパラメータ
        self.__alpha = 1.0
        # 事後分布のパラメータ
        self.__alpha_hat = np.zeros(6)
        # カテゴリ数
        self.__K = K
        # 各カテゴリの出現回数 Sn,k
        self.__snk = np.zeros(6) 
        # カテゴリ分布のパラメータ
        self.__pi = np.zeros(6)
        # 0を回避するための値
        self.__eps = 1e-7
    
    # 出現回数を計算
    def clac_category_count(self, data):
        # データ数(1次元離散データ)
        N = len(data)
        # 出現回数を計算
        for n in range(N):
            self.__snk[int(data[n])] += 1
    
    # 事後分布のパラメータを学習
    def update(self, data):
        # 出現回数を計算
        self.clac_category_count(data)
        
        # 事後分布のパラメータ
        alpha_hat = np.zeros(6)

        # 事後分布のパラメータを計算
        for k in range(self.__K):
            # 事後分布のパラメータを計算
            alpha_hat[k] = self.__snk[k] + self.__alpha

        '''
        # 更新回数
        N = 100
        # 対数尤度
        llhs = []
        # パラメータ
        pis = []
        # カテゴリ分布のパラメータの計算範囲
        x = np.arange(0, 1, 0.01)
        # 更新回数
        for n in range(N):
            llh = 0
            alpha_hat = np.zeros(6)
            pi = [x[n], (1-x[n]) * 0.2, (1-x[n]) * 0.2, (1-x[n]) * 0.2, (1-x[n]) * 0.2, (1-x[n]) * 0.2]
            for k in range(self.__K):
                # 事後分布のパラメータを計算
                alpha_hat[k] = self.__snk[k] + self.__alpha
                # 対数尤度を計算
                llh += (alpha_hat[k] - 1) * np.log(pi[k] + self.__eps)
            llhs.append(llh)
            alpha_hats.append(alpha_hat)
        '''
        # 事後分布のパラメータ
        self.__alpha_hat = alpha_hat
        print(self.__alpha_hat)

    # 予測分布 -> カテゴリ分布のパラメータを取り出す -> 出現回数から最尤推定的に決まってしまう？
    def predict(self):
        prob = np.zeros(self.__K)    
        for k in range(self.__K):
            prob[k] += self.__alpha_hat[k]/np.sum(self.__alpha_hat)
        self.__pi = prob
        print(self.__pi)


# 実行文
if __name__ == '__main__':
    cat = Categorical()
    data = [0,0,0,1,1,1,2,2,2,3,3,4,4,4,5,5,5]
    cat.update(data)
    cat.predict()