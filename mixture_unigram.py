# coding:utf-8
import numpy as np
import scipy.special
from sklearn import preprocessing
import sys
import math

def vectorize(docs, vocabularies):
    D = len(docs)
    V = len(vocabularies)
    BOWs = np.zeros([D, V])
    print("Now vectorizing...")
    for i, doc in enumerate(docs):
        sys.stdout.write("\r%d / %d" % (i+1,D))
        sys.stdout.flush()
        # まずリストに変換
        doc = doc.rstrip("\n").split(",")
        for word in doc:
            try:
                ix = vocabularies.index(word)
                BOWs[i, ix] += 1
            except:
                continue
    print("\n")
    print("Done!")
    return BOWs

class MixtureUnigram():

    def __init__(self, BOWs, K=2, V=18, alpha=1.0, beta=1.0):
        self.BOWs = BOWs # BOWになってる文書集合
        self.D = BOWs.shape[0] #文書数

        self.K = K
        self.V = V

        self.alpha = alpha
        self.beta = beta

        self.D_k = np.zeros([self.K, 1]) #トピックkを振られた文書数
        self.N_k = np.zeros([self.K, 1]) #文書集合全体でトピックkを振られた単語数
        self.N_kv = np.zeros([self.K, self.V]) # トピックkに割り振られている語彙vの数
        self.z_d = np.zeros([self.D, 1]) - 1# 文書dに割り振られてたトピック

        #self.pplx_ls = list()

    def fit(self, epoch=100):

        for e in range(epoch):
            print("\nEpoch: {}".format(e+1))
            for d, BOW in enumerate(self.BOWs):
                sys.stdout.write("\r%d / %d" % (d+1, len(self.BOWs)))
                sys.stdout.flush()

                not_zero_indices = np.where(BOW > 0)[0] # 非ゼロ要素の単語index
                current_topic = int(self.z_d[d])
                N_d = BOW.sum()

                # reset information of d-th BOW
                print("current_topic",current_topic)
                if current_topic >= 0:
                    self.D_k[current_topic] -= 1
                    for index in not_zero_indices:
                        self.N_kv[current_topic, index] -= BOW[index]
                    self.N_k[current_topic] -= N_d

                #p_z = self._calc_probability(BOW, N_d, not_zero_indices)

                p_z = np.zeros(self.K)
                for k in range(self.K):
                    first = (self.D_k[k] + self.alpha) * math.gamma(self.N_k[k]+(self.beta*self.V)) / math.gamma(self.N_k[k] + N_d + (self.beta*self.V))
                    second = 1.0
                    for index in not_zero_indices:
                        second *= math.gamma(self.N_kv[k, index] + BOW[index] + self.beta) / math.gamma(self.N_kv[k, index] + self.beta)
                    if first*second < 0:
                        raise "hogehoge"
                    p_z[k] = first*second

                #p_z = self._normalize(p_z)
                p_z = preprocessing.normalize(p_z, norm="l1")[0] # 正規化

                new_topic = self._sampling_topic(p_z)
                self.z_d[d] = new_topic

                print("\ndoc id = ", new_topic)
                for k in range(self.K):
                    print("topic",k,"'s pro =", p_z[k])

                self.D_k[new_topic] += 1
                self.N_k[new_topic] += N_d
                for index in not_zero_indices:
                    self.N_kv[new_topic, index] += BOW[index]

            # *** End d loop ***

            # update parameters

            """
            digamma = scipy.special.digamma
            a_numerator = np.sum(digamma(self.D_k+self.alpha)) - (self.K*digamma(self.alpha))
            a_denominator = self.K*(digamma(self.D+self.alpha*self.K) - digamma(self.alpha*self.K))

            b_numerator = np.sum(digamma(self.N_kv+self.beta)) - self.K*self.V*digamma(self.beta)
            b_denominator = self.V*np.sum(digamma(self.N_kv+self.beta*self.V)) - self.K*self.V*digamma(self.beta*self.V)

            self.alpha = self.alpha * a_numerator / a_denominator
            self.beta = self.beta * b_numerator / b_denominator
            """

            # update α
            numerator = 0.0
            digamma = scipy.special.digamma
            for k in range(self.K):
                numerator += digamma(self.D_k[k]+self.alpha)
            numerator -= self.K*digamma(self.alpha)
            self.alpha = self.alpha * numerator / (self.K*digamma(self.D+self.alpha*self.K)-self.K*digamma(self.alpha*self.K))
            # end update α

            # update β
            numerator = 0.0
            denominator = 0.0
            for k in range(self.K):
                for v in range(self.V):
                    numerator += digamma(self.N_kv[k,v]+self.beta)
                denominator += digamma(self.N_k[k]+self.beta*self.V)
            numerator -= self.K*self.V*digamma(self.beta)
            denominator = self.V*denominator - self.K*self.V*digamma(self.beta*self.V)
            self.beta = self.beta*numerator/denominator


            # パープレキシティの計算
            """
            perplexity = self._calc_perplexity()
            self.pplx_ls.append(perplexity)
            print(self.z_d)
            print("")
            print("perplexity:{}".format(perplexity))
            """

        print(self.z_d)
        print("\n")

    def _calc_probability(self, BOW, N_d, not_zero_indices):
        # calculate sampling porbability
        # gamma = scipy.special.gamma
        gamma = math.gamma
        p_z = np.zeros(self.K)
        for k in range(self.K):
            A = (self.D_k[k]+self.alpha) * gamma(self.N_k[k]+(self.beta*self.V))\
              / gamma(self.N_k[k]+N_d+(self.beta*self.V))
            B = 1
            for index in not_zero_indices:
                B *= gamma(self.N_kv[k, index] + BOW[index] + self.beta)\
                   / gamma(self.N_kv[k, index] + self.beta)
            if A * B  < 0:
                print(A)
                print(B)
                return
            p_z[k] = A * B
        return p_z

    def _sampling_topic(self, p_z):
        return np.argmax(np.random.multinomial(1, p_z, size=1))

    def _normalize(self, vec):
        return vec / vec.sum()

    def _calc_perplexity(self):
        # calculate perplexity
        theta_k = (self.D_k+self.alpha) / (self.D+self.alpha*self.K) #k次元ベクトル
        phi_kv = (self.N_kv+self.beta) / (self.N_k+self.beta*self.V)

        numerator = 0
        denominator = 0
        for BOW in self.BOWs:
            denominator += BOW.sum() # 単語数をカウントしていく
            likelihood = np.sum(theta_k*np.prod(phi_kv**BOW, axis=1, keepdims=True))
            if likelihood > 0:
                numerator += np.log(likelihood) #対数尤度の計算
        return np.exp(-1*numerator/denominator)

if __name__ == "__main__":
    import warnings;warnings.filterwarnings('ignore')

    docfile = sys.argv[1]
    vocfile = sys.argv[2]

    with open(docfile, "r") as f:
        docs = f.readlines()
    with open(vocfile, "r") as f:
        vocabularies = f.read().rstrip("\n").split(",")

    BOWs = vectorize(docs, vocabularies)

    model = MixtureUnigram(BOWs, K=3, V=len(vocabularies))
    model.fit(epoch=10)
