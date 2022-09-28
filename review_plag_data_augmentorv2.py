
#####   Data Augmentor

import pandas as pd
rfile = "/home/mohammadhammad/data/cr.mysql"
rData = pd.read_csv(rfile,delimiter="\t")
## remove error special char id review 2178455
FA = "fa"
PL = "pl"
IN = "in"
data = {FA:{},PL:{},IN:{}}

for index, row in rData.iterrows():
    if not row.isnull().any():  ## 34803 count remaining
        data[PL][str(row["id"])+"_"+PL] = row["pd"].replace("\\n"," ").replace("\xa0"," ")
        data[IN][str(row["id"])+"_"+IN] = row["id"].replace("\\n"," ").replace("\xa0"," ")
        data[FA][str(row["id"])+"_"+FA] = row["fd"].replace("\\n"," ").replace("\xa0"," ")


import nltk,random
from tqdm.auto import tqdm

sdata = {FA:{},PL:{},IN:{}}
for key,val in tqdm(data[FA].items()):
    sdata[FA][key] = nltk.sent_tokenize(val)

for key,val in tqdm(data[PL].items()):
    sdata[PL][key] = nltk.sent_tokenize(val)

for key,val in tqdm(data[IN].items()):
    sdata[IN][key] = nltk.sent_tokenize(val)


from collections import Counter
## max sentences
def stats(type_):
    cnt=0
    for key,val in sdata[type_].items():
        cnt = max(cnt,len(val))
    print(cnt)
    cnt=Counter()
    cnt.update([len(val) for val in sdata[type_].values()])
    print(cnt.most_common())

stats(FA)
stats(PL)
stats(IN)

# 20
# [(4, 10052), (5, 8511), (3, 6178), (6, 4532), (7, 1877), (2, 1735), (8, 755), (1, 551), (9, 271), (10, 174), (11, 70), (12, 49), (13, 16), (14, 13), (17, 6), (16, 4), (15, 4), (20, 2), (0, 1), (18, 1), (19, 1)]
# 24
# [(4, 10592), (5, 8429), (3, 7067), (6, 3534), (2, 2286), (7, 1424), (1, 542), (8, 511), (9, 187), (10, 101), (11, 55), (12, 30), (13, 19), (14, 7), (15, 6), (16, 5), (17, 4), (20, 2), (22, 1), (24, 1)]
# 36
# [(4, 8122), (5, 7772), (6, 5385), (3, 4969), (7, 2743), (2, 1867), (8, 1474), (9, 724), (1, 605), (10, 415), (11, 246), (12, 155), (13, 89), (14, 74), (15, 42), (16, 40), (17, 17), (18, 14), (19, 10), (21, 10), (20, 8), (22, 7), (25, 4), (26, 2), (27, 2), (35, 1), (32, 1), (36, 1), (29, 1), (30, 1), (23, 1), (24, 1)]

### upto 7 sentences seem fine, others can be trimmed here itself to avoid Bert 512 word 


## max words, for bert,
## ignoring as https://github.com/UKPLab/sentence-transformers/issues/574
def stats_words(type_):
    cnt=0
    cnt_=Counter()
    for key,val in tqdm(sdata[type_].items()):
        # print(val)
        len_ = [ len(nltk.word_tokenize(snt)) for snt in val]
        cnt = max(cnt,sum(len_))
        cnt_.update([sum(len_)])
    print(cnt)
    print(cnt_.most_common())

stats_words(FA)
stats_words(PL)
stats_words(IN)

# 399
# 376
# 523

### slight info loss tobe chunked later on:
## ignoring as https://github.com/UKPLab/sentence-transformers/issues/574

# def truncate(type_):
#     MAX_SENT = 7
#     for key,val in tqdm(data[type_].items()):
#         if len(sdata[type_][key]) > MAX_SENT:
#             sdata[type_][key] = sdata[type_][key][0:MAX_SENT]

# truncate(FA)
# stats_words(FA) #276
# truncate(PL)
# stats_words(PL) #233
# truncate(IN)
# stats_words(IN) #280



# for key,val in tqdm(sdata[type_].items()):

##### Ad shuffled DATA ???? imp with 0 10 % ##########

def augment(sdata,type_):
    revList = list(sdata[type_].items())
    revlen = len(revList)
    stat_aug = Counter()
    buckets = Counter()
    sentence_uni = set()
    for row in revList:
        sentence_uni.update(row[1])
    sentence_uni = list(sentence_uni)
    finalData = []
    for i,(key, sents) in tqdm(enumerate(revList)):
        if i < int(0.09*revlen): # first 10%
            ## different
            buckets.update([int(0.09*revlen)])
            finalData.append([key," ".join(random.sample(revList,1)[0][1])," ".join(sents),0])
        elif i < int(0.14*revlen):
            # just reshuffled, 0.9
            buckets.update([int(0.14*revlen)])
            finalData.append([key," ".join(random.sample(sents,len(sents)))," ".join(sents),0.9])
        elif i < int(0.19*revlen):
            # same
            buckets.update([int(0.19*revlen)])
            finalData.append([key," ".join(sents)," ".join(sents),1])
        elif i < int(0.28*revlen):
            # add atleast max(1,int(20%sent of orig rev)) rand sentences, at rand pos,0.9
            SCORE = 0.9 
            rand_add_cnt = max(1,int(0.2*len(sents)))
            rand_sents = random.sample(sentence_uni,rand_add_cnt)
            orig_rev = list(sents)
            for i in range(len(rand_sents)):
                orig_rev.insert(random.randint(0, len(orig_rev)), rand_sents[i])
            buckets.update([int(0.28*revlen)])
            finalData.append([key," ".join(orig_rev)," ".join(sents),SCORE])
            stat_aug.update(["rand_add_cnt: "+str(rand_add_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])
        elif i < int(0.37*revlen):
            # add atleast max(2,int(40%sent)) rand, at rand pos,0.8 
            SCORE = 0.8 
            rand_add_cnt = max(2,int(0.4*len(sents)))
            rand_sents = random.sample(sentence_uni,rand_add_cnt)
            orig_rev = list(sents)
            for i in range(len(rand_sents)):
                orig_rev.insert(random.randint(0, len(orig_rev)), rand_sents[i])
            buckets.update([int(0.37*revlen)])
            finalData.append([key," ".join(orig_rev)," ".join(sents),SCORE])
            stat_aug.update(["rand_add_cnt: "+str(rand_add_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])
        elif i < int(0.46*revlen):
            # add atleast max(3,int(60%sent)) rand, at rand pos,0.7
            SCORE = 0.7 
            rand_add_cnt = max(3,int(0.6*len(sents)))
            rand_sents = random.sample(sentence_uni,rand_add_cnt)
            orig_rev = list(sents)
            for i in range(len(rand_sents)):
                orig_rev.insert(random.randint(0, len(orig_rev)), rand_sents[i])
            buckets.update([int(0.46*revlen)])
            finalData.append([key," ".join(orig_rev)," ".join(sents),SCORE])
            stat_aug.update(["rand_add_cnt: "+str(rand_add_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)]) 
        elif i < int(0.55*revlen):
            # remove from orig atmost max(1,int(25%sent)), add 3 rand, at rand pos,0.6
            SCORE = 0.6 
            rand_rem_cnt = max(1,int(0.25*len(sents)))
            rand_add_cnt = 3
            rand_sents = random.sample(sentence_uni,rand_add_cnt)
            orig_rev = list(sents)
            for _ in range(rand_rem_cnt):
                # if len(orig_rev)>0:
                orig_rev.remove(random.sample(orig_rev,1)[0])
            for i in range(len(rand_sents)):
                orig_rev.insert(random.randint(0, len(orig_rev)), rand_sents[i])
            buckets.update([int(0.55*revlen)])
            finalData.append([key," ".join(orig_rev)," ".join(sents),SCORE])
            stat_aug.update(["rand_add_cnt: "+str(rand_add_cnt)+"- rand_rem_cnt: "+str(rand_rem_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])  
        elif i < int(0.64*revlen):
            # remove from orig atmost max(2,int(50%sent)), add num removed rand, at rand pos,0.5
            SCORE = 0.5 
            rand_rem_cnt = max(2,int(0.5*len(sents)))
            if rand_rem_cnt >= len(sents):
                if len(sents) == 1:
                    print("bypassing"+str((i,key)))
                    continue ### 
                else:
                    rand_rem_cnt = max(1, len(sents)-1)
            rand_add_cnt = rand_rem_cnt
            rand_sents = random.sample(sentence_uni,rand_add_cnt)
            orig_rev = list(sents)
            for _ in range(rand_rem_cnt):
                # if len(orig_rev)>0:
                orig_rev.remove(random.sample(orig_rev,1)[0])
            for i in range(len(rand_sents)):
                orig_rev.insert(random.randint(0, len(orig_rev)), rand_sents[i])
            buckets.update([int(0.64*revlen)])
            finalData.append([key," ".join(orig_rev)," ".join(sents),SCORE])
            stat_aug.update(["rand_add_cnt: "+str(rand_add_cnt)+"- rand_rem_cnt: "+str(rand_rem_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])  
        elif i < int(0.73*revlen):
            # remove from orig atmost max(3,int(75%sent)), add num removed rand, at rand pos,0.4 
            SCORE = 0.3 
            rand_rem_cnt = max(3,int(0.75*len(sents)))
            if rand_rem_cnt >= len(sents):
                if len(sents) == 1:
                    print("bypassing"+str((i,key)))
                    continue ### 
                else:
                    rand_rem_cnt = max(1, len(sents)-1)
            rand_add_cnt = rand_rem_cnt
            rand_sents = random.sample(sentence_uni,rand_add_cnt)
            orig_rev = list(sents)
            for _ in range(rand_rem_cnt):
                # if len(orig_rev)>0:
                orig_rev.remove(random.sample(orig_rev,1)[0])
            for i in range(len(rand_sents)):
                orig_rev.insert(random.randint(0, len(orig_rev)), rand_sents[i])
            buckets.update([int(0.73*revlen)])
            finalData.append([key," ".join(orig_rev)," ".join(sents),SCORE])
            stat_aug.update(["rand_add_cnt: "+str(rand_add_cnt)+"- rand_rem_cnt: "+str(rand_rem_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])  
        elif i < int(0.82*revlen):
            # max(1, int(30%)) actual in a rand rev, at rand pos,0.25 
            SCORE = 0.2 
            actual_retain_cnt = max(1,int(0.3*len(sents)))
            actual_rev = random.sample(sents,actual_retain_cnt)
            rand_rev = random.sample(revList,1)[0][1]
            rand_rev_cnt = len(rand_rev)
            for i in range (len(actual_rev)):
                rand_rev.insert(random.randint(0, len(rand_rev)), actual_rev[i])
            buckets.update([int(0.82*revlen)])
            finalData.append([key," ".join(rand_rev)," ".join(sents),SCORE])
            stat_aug.update(["actual_retain_cnt: "+str(actual_retain_cnt)+"- rand_rev_cnt: "+str(rand_rev_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])  
        elif i < int(0.91*revlen):
            # max(1,int(20%)) actual in a rand rev, at rand pos,0.1
            SCORE = 0.1 
            actual_retain_cnt = max(1,int(0.2*len(sents)))
            actual_rev = random.sample(sents,actual_retain_cnt)
            rand_rev = random.sample(revList,1)[0][1]
            rand_rev_cnt = len(rand_rev)
            for i in range (len(actual_rev)):
                rand_rev.insert(random.randint(0, len(rand_rev)), actual_rev[i])
            buckets.update([int(0.91*revlen)])
            finalData.append([key," ".join(rand_rev)," ".join(sents),SCORE])
            stat_aug.update(["actual_retain_cnt: "+str(actual_retain_cnt)+"- rand_rev_cnt: "+str(rand_rev_cnt)+"/"+str(len(sents))+" s-"+str(SCORE)])
        elif i < int(1*revlen):
            # NEW CASE: random reshuffle words
            SCORE = 0
            orig_rev = " ".join(sents)
            shuffled_rev = random.sample(orig_rev.split(),len(orig_rev.split()))
            buckets.update([int(1*revlen)])
            finalData.append([key," ".join(shuffled_rev),orig_rev,SCORE])
    print(stat_aug.most_common())
    print(buckets.most_common())
    return finalData

augmented = augment(sdata,PL)+augment(sdata,FA)+augment(sdata,IN)
len(augmented) ## 104097
random.shuffle(augmented)

import pickle
with open('datav2.pickle', 'wb') as handle:
    pickle.dump(augmented, handle, protocol=pickle.DEFAULT_PROTOCOL)

with open('/home/mohammadhammad/data/datav2.pickle', 'rb') as handle:
    augmented = pickle.load(handle)

### cases not working = \\n
# raw_text = 'Faculty are good but management is not good.\\nExams are online due to covid but final exam will be offline, some faculty members are good in nature and friendly but some are not.'




# from spacy.lang.en import English 
# nlp = English()
# nlp.add_pipe('sentencizer') 
# # nlp.add_pipe(nlp.create_pipe('sentencizer')) 
# doc = nlp(raw_text)
# sentences = [sent.string.strip() for sent in doc.sents]




import pickle
rfile = "/home/mohammadhammad/data/datav2.pickle"
# rfile = "datav2.pickle"
with open(rfile, 'rb') as handle:
    train = pickle.load(handle)

train = [[row[1],row[2],row[3]] for row in train]
test = train[-4000:] ## already shuffled
train = train[0:len(train)-4000]
val = train[-10000:] 
train = train[0:len(train)-10000]
len(train),len(val),len(test)


