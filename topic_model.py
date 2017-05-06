# coding:utf-8
import numpy as np
import scipy.special
from sklearn import preprocessing
import sys
import pandas as pd


digamma = scipy.special.digamma

vocfile = sys.argv[1]
with open(vocfile, "r") as f:
    vocabularies = f.read().rstrip("\n").split(",")

def flatten(BOWs):
    # transform vector
    new_BOWs = np.zeros([7343, 5000], dtype=np.int16) - 1
    for d, BOW in enumerate(BOWs):

        sys.stdout.write("\r%d / %d" % (d+1, len(BOWs)))
        sys.stdout.flush()

        ix = 0
        print(BOW)
        not_zero_indices = np.where(BOW > 0)[0]
        print(not_zero_indices)
        for v in not_zero_indices:
            print(v)
            freq = int(BOW[v])
            for f in range(freq):
                new_BOWs[d, ix] = v
                ix += 1
    print("\n")
    print("done!")
    return new_BOWs


class TopicModel():

    def __init__(self, BOWs, K=20, V=5000, max_words=2000, ratio=0.9 ,alpha=1.0, beta=1.0):
        self.BOWs = BOWs
        border = int(ratio * self.BOWs.shape[0])

        self.train_BOWs, self.test_BOWs = np.vsplit(self.BOWs, [border])

        self.V = V
        self.K = K

        self.alpha = alpha
        self.beta  = beta

        self.D = self.train_BOWs.shape[0] #学習に使う文書数
        self.test_D = self.test_BOWs.shape[0]

        self.N_dk = np.zeros([self.D, self.K]) #トピックkを振られた文書数
        self.N_kv = np.zeros([self.K, self.V]) # トピックkに割り振られている語彙vの数
        self.N_k  = np.zeros([self.K, 1]) #文書集合全体でトピックkを振られた単語数

        self.z_dn = np.zeros([self.D, max_words]) - 1 # 文書dのn番目の単語に割り振られてたトピック

    def fit(self, epoch=100):

        self.pplx_ls = np.zeros([epoch])

        for e in range(epoch):
            print("Epoch: {}".format(e+1))

            for d, BOW in enumerate(self.train_BOWs):
                sys.stdout.write("\r%d / %d" % (d+1, self.train_BOWs.shape[0]))
                sys.stdout.flush()

                for n, v in enumerate(BOW):
                    if v < 0: break

                    current_topic = int(self.z_dn[d, n])

                    # reset information of d-th BOW
                    if current_topic >= 0:
                        self.N_dk[d, current_topic] -= 1
                        self.N_kv[current_topic, v] -= 1
                        self.N_k[current_topic] -= 1

                    # sampling
                    p_z_dn = self._calc_probability(d, v)
                    new_topic = self._sampling_topic(p_z_dn)
                    self.z_dn[d, n] = new_topic

                    # update counting
                    self.N_dk[d, new_topic] += 1
                    self.N_kv[new_topic, v] += 1
                    self.N_k[new_topic] += 1


            # update α
            numerator = np.sum(digamma(self.N_dk+self.alpha))\
                      - self.D*self.K*digamma(self.alpha)
            denominator = self.K*(np.sum(digamma(np.count_nonzero(self.train_BOWs+1,axis=1)+self.alpha*self.K))\
                        - self.D*digamma(self.alpha*self.K))
            self.alpha *= numerator / denominator

            # update β
            numerator = np.sum(digamma(self.N_kv+self.beta)) - self.K*self.V*digamma(self.beta)
            denominator = self.V*(np.sum(digamma(self.N_k+self.beta*self.V)) - self.K*digamma(self.beta*self.V))
            self.beta *= numerator / denominator

            print("\nparameters")
            print("alpha :{}".format(self.alpha))
            print("beta :{}".format(self.beta))

            df = pd.DataFrame(self.N_kv, columns=vocabularies)
            topic1 = pd.DataFrame(df.ix[0], columns=["topic1"])
            topic2 = pd.DataFrame(df.ix[1], columns=["topic2"])
            print("---------------------")
            print(topic1.sort_values(by=["topic1"], ascending=False).head(10))
            print("---------------------")
            print(pd.DataFrame(df.ix[1]).sort_values(by=[1], ascending=False).head(10))
            print("*********************")

    def _calc_probability(self, d, v):
        # calculate sampling porbability
        p_z_dn = np.zeros(self.K)
        for k in range(self.K):
            A = self.N_dk[d,k] + self.alpha
            B = (self.N_kv[k,v] + self.beta) / (self.N_k[k] + self.beta*self.V)
            p = A * B
            if p  < 0:
                raise "probabirity is less than zero!"
            p_z_dn[k] = p
        p_z_dn = preprocessing.normalize(p_z_dn, norm="l1")[0] # 正規化
        return p_z_dn

    def _sampling_topic(self, p_z_dn):
        return np.argmax(np.random.multinomial(1, p_z_dn, size=1))

    def _calc_perplexity(self):
        print("\ncalculating perplexity...")

        test_N_dk = np.zeros([self.test_D, self.K])
        test_N_kv = np.zeros([self.K, self.V])
        test_N_k = np.zeros([self.K, 1])

        # predict
        for d, BOW in enumerate(self.test_BOWs):
            sys.stdout.write("\r%d / %d" % (d+1, self.test_BOWs.shape[0]))
            sys.stdout.flush()
            for v in BOW:
                if v < 0: break
                p_z_dn = self._calc_probability(d, v)
                predict_topic = self._sampling_topic(p_z_dn)
                test_N_dk[d, predict_topic] += 1
                test_N_kv[predict_topic, v] += 1
                test_N_k[predict_topic] += 1

        # calculate perplexity
        numerator = 0
        denominator = 0
        for BOW in self.test_BOWs:
            N_d = np.count_nonzero(BOW+1)
            theta_k = (test_N_dk+self.alpha) / (N_d+self.alpha*self.K) # K-dim vector
            phi_kv = (test_N_kv+self.beta) / (test_N_k+self.beta*self.V) # K*V-dim matrix
            likelihood = np.prod(np.sum(theta_k*phi_kv, axis=0))
            numerator += np.log(likelihood)
            denominator += N_d
        return np.exp(-1*numerator/denominator)

if __name__ == "__main__":
    import warnings;warnings.filterwarnings('ignore')
    BOWs = np.load("./wikidata.npy")
    model = TopicModel(BOWs, ratio=1.0)
    model.fit(epoch=100)
    np.save("output", model.N_kv)
