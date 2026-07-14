from scipy.stats import norm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math
import tqdm
from scipy.spatial.distance import pdist
from .verification import evaluate, evaluate_binary


def adduserspecfkey(embeddings_nonce,issame, worstcase=False):
    embeddings_nonce_xorkey = np.zeros(embeddings_nonce.shape)
    totl = int(embeddings_nonce.shape[0]/2)
    dim = embeddings_nonce.shape[1]
    for i in range(totl):
        if not issame[i]:
            nonce1 = np.random.random(size=dim)-0.5>0
            if worstcase:
                nonce2 = nonce1
            else:
                nonce2 = np.random.random(size=dim)-0.5>0
                
            embeddings_nonce_xorkey[2*i,:] = np.logical_xor(embeddings_nonce[2*i,:],nonce1)
            embeddings_nonce_xorkey[2*i+1,:] = np.logical_xor(embeddings_nonce[2*i+1,:],nonce2)
        else:
            nonce1 = np.random.random(size=dim)-0.5>0
            embeddings_nonce_xorkey[2*i,:] = np.logical_xor(embeddings_nonce[2*i,:],nonce1)
            embeddings_nonce_xorkey[2*i+1,:] = np.logical_xor(embeddings_nonce[2*i+1,:],nonce1)
    return embeddings_nonce_xorkey

def equal_probable(embeddings_o, intervalnum=4):
    meanv,stdv = np.mean(np.sum(embeddings_o,axis=1)/512), np.mean(np.std(embeddings_o,axis=1))
    print('mean: ',meanv,', std:', stdv)
    minv = np.min(embeddings_o)
    maxv = np.max(embeddings_o)
    startv = minv - 0.1
    interval = 1 / intervalnum
    intervals = []
    for i in range(intervalnum-1):
        for endv in np.arange(startv, maxv + 0.1, 0.0001):
            pro=norm(meanv, stdv).cdf(endv) - norm(meanv, stdv).cdf(startv)
            if pro>=interval:
                intervals.append(endv)
                startv = endv
                break
    assert len(intervals) == intervalnum-1
    return np.array(intervals)

def equal_space(embeddings_o, intervalnum=4):
    minv = np.min(embeddings_o)
    maxv = np.max(embeddings_o)
    step = (maxv-minv)/intervalnum
    intervals = []
    for i in range(1,intervalnum):
        intervals.append(minv+step*i)
    return np.array(intervals)

def onehot_binary(embeddings_o,interval = np.array([-0.03, 0 , 0.03])):
    lkut = np.zeros((len(interval)+1,len(interval)+1)) 
    for i in range(len(interval)+1):
        lkut[i,len(interval)-i] = 1
    print(lkut)
    block = len(interval) + 1
    def LSSC(data):
        new_data = np.zeros(512*block)
        for i in range(len(data)):
            index = -1
            whereindex = np.where(interval>data[i])
            if len(whereindex[0]) != 0:
                index = whereindex[0][0]
            new_data[i*block:(i+1)*block] = lkut[index]
        return new_data

    embeddings = np.zeros((len(embeddings_o),512*block))
    for i in tqdm.tqdm(range(len(embeddings_o))):
        embeddings[i,:] =  LSSC(embeddings_o[i,:])
    
    return embeddings

def lssc_binary(embeddings_o,interval = np.array([-0.03, 0 , 0.03])):
    lkut = np.zeros((len(interval)+1,len(interval))) 
    for i in range(1,len(interval)+1):
        lkut[i,len(interval)-i:] = 1
    def LSSC(data):
        new_data = np.zeros(512*len(interval))
        for i in range(len(data)):
            index = -1
            whereindex = np.where(interval>data[i])
            if len(whereindex[0]) != 0:
                index = whereindex[0][0]
            new_data[i*len(interval):(i+1)*len(interval)] = lkut[index]
        return new_data
    block = len(interval) 

    embeddings = np.zeros((len(embeddings_o),512*block))
    for i in tqdm.tqdm(range(len(embeddings_o))):
        embeddings[i,:] =  LSSC(embeddings_o[i,:])
    return embeddings

def brgc_binary(embeddings_o,interval = np.array([-0.03, 0 , 0.03])):
    block = 2
    lkut = np.zeros((len(interval)+1,block)) 
    for i in range(1,len(interval)+1):
        lkut[i,:] = np.array( [int(s) for s in "{0:02b}".format(i)])

    print(lkut)

    def LSSC(data):
        new_data = np.zeros(512*block)
        for i in range(len(data)):
            index = -1
            whereindex = np.where(interval>data[i])
            if len(whereindex[0]) != 0:
                index = whereindex[0][0]
            # print(index,embeddings_o[0,i], np.where(interval>embeddings_o[0,i]),interval>embeddings_o[0,i])        
            new_data[i*block:(i+1)*block] = lkut[index]
        return new_data

    embeddings = np.zeros((12000,512*block))
    for i in tqdm.tqdm(range(12000)):
        embeddings[i,:] =  LSSC(embeddings_o[i,:])
    return embeddings

def d_prime_savefig(embeddings,issame,dist='hamming', plot=''):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    def get_cos_similar(v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    if dist=='hamming':
        dist = np.logical_xor(emb1,emb2) * 1
        bio_noise = np.logical_xor(emb1,emb2) * 1
        gen = bio_noise[issame==1]
        imp = bio_noise[issame==0]
        gen = np.sum(gen,axis = 1)/embeddings.shape[1]
        imp = np.sum(imp,axis = 1)/embeddings.shape[1]
        Dprime= np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen*100, bins=100, label="Mated: {:.2%}".format(np.mean(gen)))
            plt.hist(imp*100, bins=100, label="Nonmated: {:.2%}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Hamming distance in percentage')
            plt.ylabel('Frequency')
            plt.show()
            fig.savefig(plot)
    elif dist=='cosine':
        dist = []
        for i in range(len(emb1)):
            dist.append(1-get_cos_similar(emb1[i,:], emb2[i,:]))
        dist = np.array(dist)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Cosine distance')
            plt.ylabel('Frequency')
            plt.show()
            fig.savefig(plot)

    else:
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequency')
            plt.show()
            fig.savefig(plot)

    
    return Dprime, gen, imp

def d_prime(embeddings,issame,dist='hamming', plot=''):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    def get_cos_similar(v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    if dist=='hamming':
        dist = np.logical_xor(emb1,emb2) * 1
        bio_noise = np.logical_xor(emb1,emb2) * 1
        gen = bio_noise[issame==1]
        imp = bio_noise[issame==0]
        gen = np.sum(gen,axis = 1)/embeddings.shape[1]
        imp = np.sum(imp,axis = 1)/embeddings.shape[1]
        Dprime= np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen*100, bins=100, label="Mated: {:.2%}".format(np.mean(gen)))
            plt.hist(imp*100, bins=100, label="Nonmated: {:.2%}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Hamming distance in percentage')
            plt.ylabel('Frequency')
            plt.show(block=False)
            fig.savefig(plot)
    elif dist=='cosine':
        dist = []
        for i in range(len(emb1)):
            dist.append(1-get_cos_similar(emb1[i,:], emb2[i,:]))
        dist = np.array(dist)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Cosine distance')
            plt.ylabel('Frequency')
            plt.show(block=False)
            fig.savefig(plot)

    else:
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequency')
            plt.show(block=False)
            fig.savefig(plot)

    
    return Dprime, gen, imp

def computeGenImp(embeddings,issame,dist='hamming', plot=''):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    def get_cos_similar(v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    if dist=='hamming':
        dist = np.logical_xor(emb1,emb2) * 1
        bio_noise = np.logical_xor(emb1,emb2) * 1
        gen = bio_noise[issame==1]
        imp = bio_noise[issame==0]
        gen = np.sum(gen,axis = 1)/embeddings.shape[1]
        imp = np.sum(imp,axis = 1)/embeddings.shape[1]
       
    elif dist=='cosine':
        dist = []
        for i in range(len(emb1)):
            dist.append(1-get_cos_similar(emb1[i,:], emb2[i,:]))
        dist = np.array(dist)
        gen = dist[issame==1]
        imp = dist[issame==0]

    else:
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1)
        gen = dist[issame==1]
        imp = dist[issame==0]
  
    
    return gen, imp


def addNonce(embeddings, crossover_prob=0.30, seed=None):
    """
    Apply a random AND mask (nonce) to binary embeddings.

    For each bit position, the nonce bit is 1 with probability (1 - crossover_prob)
    and 0 with probability crossover_prob.  After masking, all positions where the
    nonce is 0 contribute zero bits regardless of the biometric — eliminating that
    fraction of the intra-class Hamming noise.

    This is the paper's mask r = sgn(u − kappa), with crossover_prob = kappa.

    Parameters
    ----------
    embeddings     : (N, D) binary array
    crossover_prob : fraction of mask bits that are 0  (= kappa in the paper)
    seed           : optional RNG seed for reproducibility

    Returns
    -------
    embeddings_nonce : masked binary array, same shape as embeddings
    nonce            : (D,) binary mask that was applied
    """
    if seed is not None:
        np.random.seed(seed)
    nonce = np.random.random(size=embeddings.shape[1]) - crossover_prob > 0
    embeddings_nonce = np.zeros(embeddings.shape)
    for i in range(len(embeddings)):
        embeddings_nonce[i, :] = np.logical_and(embeddings[i, :], nonce)
    return embeddings_nonce, nonce


def look4noncerate_joint(
    embeddings,
    issame,
    gen_threshold=0.15,
    imp_threshold=0.20,
    gen_confidence=0.95,
    imp_confidence=1.0,
    step=0.005,
):
    """
    Smallest crossover_prob (= κ) such that both:
      - genuine pairs: masked Hamming BER <= gen_threshold (fraction >= gen_confidence)
      - impostor pairs: masked Hamming BER >= imp_threshold (fraction >= imp_confidence)

    Scanning κ upward from 0 yields the minimum κ that still separates impostors — this
    keeps as many mask-1 bits as possible while pushing masked impostor distance up,
    which typically reduces LDPC false accepts (lower FAR) at some FRR cost.

    Returns (None, None, None) if no κ in [0, 0.99) satisfies both constraints.
    """
    for crossover_prob in tqdm.tqdm(
        np.arange(0.0, 0.99, step),
        desc="Joint κ (genuine + impostor)",
    ):
        embeddings_nonce, nonce = addNonce(embeddings, crossover_prob=crossover_prob)
        gens, imps = computeGenImp(embeddings_nonce, issame, dist="hamming")
        if len(gens) == 0:
            continue
        ok_gen = np.sum(gens <= gen_threshold) / len(gens) >= gen_confidence
        if len(imps) == 0:
            ok_imp = True
        else:
            ok_imp = np.sum(imps >= imp_threshold) / len(imps) >= imp_confidence
        if ok_gen and ok_imp:
            tpr, fpr, accuracy, best_thresholds = evaluate_binary(
                embeddings_nonce, issame, 10
            )
            print(
                f"joint crossover_prob: {crossover_prob:.4f}, "
                f"accuracy: {accuracy.mean():.4f} ± {accuracy.std():.4f}"
            )
            return embeddings_nonce, crossover_prob, nonce
    print("look4noncerate_joint: no κ satisfied both constraints; relax imp_threshold or gen_confidence.")
    return None, None, None


def look4noncerate(embeddings, issame, threshold=0.176, genorimp=1,
                   confidence=0.95, start=0.99):
    """
    Search for the minimum crossover_prob (= kappa) such that the target
    fraction of genuine pairs has masked BER within *threshold*.

    This is the paper's Eq. 5 calibration procedure (Section IV-D).

    Parameters
    ----------
    embeddings    : (2N, D) pair-interleaved binary array (after M_matrix + LSSC).
                    Use the same D as bits entering the LDPC path at runtime (e.g. first 832
                    of the flattened LSSC vector when the handler uses b_masked[:832]).
    issame        : (N,) int/bool — 1 = genuine pair, 0 = impostor
    threshold     : BER tolerance target (default 0.176 = Neural-MS Z=10 limit)
    genorimp      : 1 = optimise for genuine TAR (find minimum kappa);
                    0 = optimise for impostor FAR (find maximum kappa)
    confidence    : fraction of pairs that must satisfy the BER criterion
    start         : starting value when searching downward (genorimp=0)

    Returns
    -------
    embeddings_nonce : masked embeddings at the found crossover_prob
    crossover_prob   : optimal kappa
    nonce            : the binary mask used
    """
    if genorimp == 1:
        for crossover_prob in tqdm.tqdm(np.arange(0.0, 0.99, 0.005),
                                        desc="Searching kappa (genuine)"):
            embeddings_nonce, nonce = addNonce(embeddings, crossover_prob=crossover_prob)
            gens, imps = computeGenImp(embeddings_nonce, issame, dist='hamming')
            if np.sum(gens <= threshold) / len(gens) >= confidence:
                break
    else:
        for crossover_prob in tqdm.tqdm(np.arange(start, 0.0, -0.005),
                                        desc="Searching kappa (impostor)"):
            embeddings_nonce, nonce = addNonce(embeddings, crossover_prob=crossover_prob)
            gens, imps = computeGenImp(embeddings_nonce, issame, dist='hamming')
            if np.sum(imps >= threshold) / len(imps) >= confidence:
                break

    tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_nonce, issame, 10)
    print(f'crossover_prob: {crossover_prob:.4f}, '
          f'accuracy: {accuracy.mean():.4f} ± {accuracy.std():.4f}')
    return embeddings_nonce, crossover_prob, nonce
