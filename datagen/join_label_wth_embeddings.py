import os
os.environ['PYSPARK_PYTHON'] = 'python2.7'
from pyspark import SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ASpark").enableHiveSupport().getOrCreate()
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
import pandas as pd
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import random
import pickle
random.seed(0)
import seaborn as sns
import matplotlib.pyplot as plt


# Specify Parameters for the Spark Session Configurations
default_parallelism = "32"
max_parallelism = "128"
shuffle_partitions = "1024"
extraJavaOptions = [
    "-verbose:gc",
    "-XX:+PrintGCDetails",
    "-XX:+PrintGCDateStamps",
    "-XX:+UseParNewGC",
    "-XX:+UseConcMarkSweepGC",
    "-XX:+CMSConcurrentMTEnabled",
    "-XX:+OptimizeStringConcat",
    "-XX:+UseStringCache",
    "-XX:-UseGCOverheadLimit",
    "-XX:StringTableSize=24000",
    "-Djsse.enableSNIExtension=false",
    "-XX:+RelaxAccessControlCheck",
    "-Djava.net.preferIPv4Stack=true"
]
# Configurations for Spark Session
conf = SparkConf().setAll([
                       ("spark.sql.crossJoin.enabled", "true"), \
                       ("spark.rpc.message.maxSize", "2047"), \
                       ("spark.buffer.pageSize", "8m"), \
                       ("spark.cores.max", max_parallelism), \
                       ("spark.driver.memory", "8192K"), \
                       ("spark.executor.memory", "8192K"), \
                       ("spark.default.parallelism", default_parallelism), \
                       ("spark.driver.maxResultSize", "0"), \
                       ("spark.executor.extraJavaOptions", " ".join(extraJavaOptions)), \
                       ("spark.io.compression.codec", "lz4"), \
                       ("spark.io.maxRetries", "5"), \
                       ("spark.mesos.coarse", "true"), \
                       ("spark.mesos.executor.memoryOverhead", "1280"), \
                       ("spark.network.timeout", "14400"), \
                       ("spark.rdd.compress", "true"), \
                       ("spark.rpc.askTimeout", "3600"), \
                       ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"), \
                       ("spark.shuffle.compress", "true"), \
                       ("spark.sql.shuffle.partitions", shuffle_partitions), \
                       ("spark.shuffle.compress", "true"), \
                       ("spark.shuffle.spill.compress", "true"), \
                       ("spark.shuffle.io.numConnectionsPerPeer", "2"), \
                       ("spark.shuffle.file.buffer", "1024K"), \
                       ("spark.yarn.executor.memoryOverhead", "8192K"), \
                       ("spark.sql.crossJoin.enabled", "true")
                   ])

# Create a Spark Session
sparkSession = SparkSession.builder \
                           .appName("CTL_datagen") \
                           .config(conf=conf) \
                           .enableHiveSupport() \
                           .getOrCreate()


MODE = "test"
mySchema = StructType([ StructField("asin", StringType(), True)\
                       ,StructField("categories", StringType(), True)\
                       ,StructField("title", StringType(), True)\
                       ,StructField("price", StringType(), True)\
                       ,StructField("salesRank", StringType(), True)\
                       ,StructField("imUrl", StringType(), True)\
                       ,StructField("brand", StringType(), True)\
                       ,StructField("related", StringType(), True)\
                       ,StructField("description", StringType(), True)])                                                               

meta_df = spark.read.json("/user/m0m02d5/amazon/meta_Clothing_Shoes_and_Jewelry.json",\
                         schema=mySchema)\
          .select("asin", 'categories', 'title','imUrl')

meta_df = meta_df.withColumn('title', regexp_replace('title', '\n', ' '))
meta_df = meta_df.dropna(subset=('asin','categories','title','imUrl')).distinct()

# path = '/p13n_mnt/m0m02d5/amazon/image_features_Clothing_Shoes_and_Jewelry.b'

# def readImageFeatures_todict(path):
#     import array
#     f = open(path, 'rb')
#     while True:
#         asin = f.read(10)
#         if asin == '': break
#         a = array.array('f')
#         a.fromfile(f, 4096)
#         yield {'asin':asin, 'iev':a.tolist()}

# def getDF(path):
#     df = {}
#     schema = StructType([
#         StructField("asin", StringType(), True), 
#         StructField("iev", StringType(), False)])    
#     p_df = spark.createDataFrame([],schema)    
#     for idx,d in enumerate(readImageFeatures_todict(path)):
#         new_row = spark.createDataFrame([(d['asin'], d['iev'])], schema=schema)
#         p_df = p_df.union(new_row)
#     return p_df

# img_df = getDF(path)

# img_meta_df = img_df.join(mata_df, mata_df.asin == img_df.asin)\
#              .select(mata_df.asin, 'categories', 'title','imUrl','description')

#folder = 'amazon'
# meta_df.write\
#         .format("com.databricks.spark.csv")\
#         .mode("overwrite")\
#         .option("delimiter", "\01")\
#         .save(folder +"/meta_Clothing_Shoes_and_Jewelry")

'''
###############################
# Create dict which maps category: setof item ids
###############################
#Uncomment this for first time run
#Otherwise load from pickle file

meta_df_categories = meta_df.select('asin', 'categories').collect()
from collections import Counter
from collections import defaultdict
import re

tree_path_dict = defaultdict(set)
for i in range(len(meta_df_categories)): 
    t1 = meta_df_categories[i]['categories'].encode('ascii','ignore')[2:-1]
    t2 = re.sub(',\[', '', t1)
    cat_arr = t2.split(']')
    for path in cat_arr:
        #tree_path_dict_cnt[path] += 1
        tree_path_dict[path].add(meta_df_categories[i]['asin']) 
        
tree_path_dict.pop('')

f = open("/home/m0m02d5/data/amazon/Clothing_Shoes_and_Jewelry/tree_path_dict.pkl","wb")
pickle.dump(tree_path_dict,f)
f.close()
'''
with open('/home/m0m02d5/data/amazon/Clothing_Shoes_and_Jewelry/tree_path_dict.pkl', 'rb') as handle:
    tree_path_dict = pickle.load(handle)

#tree_path_dict_cnt.pop('')
mySchema = StructType([ StructField("item_id_1", StringType(), True)\
                       ,StructField("item_id_2", StringType(), True)\
                       ,StructField("label", StringType(), True)])
if MODE=="train":
    path = "amazon/Clothing_Shoes_and_Jewelry/"
    train_df =  spark.read.option("sep","\t")\
                    .schema(mySchema)\
                    .csv('/user/m0m02d5/dyadic/train.parsed.txt')
else:
    path = "amazon/Clothing_Shoes_and_Jewelry/test"
    train_df =  spark.read.option("sep","\t")\
                    .schema(mySchema)\
                    .csv('/user/m0m02d5/dyadic/test.parsed.txt')

comp_pairs = train_df.filter(train_df.label=='1').select('item_id_1', 'item_id_2')
neg_pairs = train_df.filter(train_df.label=='0').select('item_id_1', 'item_id_2')

# Find similar item to first item in complementary pair
comp_pairs_lst = comp_pairs.select('item_id_1')\
    .join(meta_df, meta_df.asin == comp_pairs.item_id_1)\
    .select('item_id_1',meta_df.categories)\
    .collect()

comp_pairs_lst2 = [None]*len(comp_pairs_lst)
comp_pairs_lst_1 = [None]*len(comp_pairs_lst)
faulty_id = []

# TODO: Put conditions that sampled items are not equal
for i, row in enumerate(comp_pairs_lst):
    catagories = comp_pairs_lst[i]['categories'].encode('ascii','ignore')[2:-1]
    
    t2 = re.sub(',\[', '', catagories)
    cat_arr = t2.split(']')
    cat_arr.remove('')
    random.shuffle(cat_arr) 
    try:
        l = list(tree_path_dict[cat_arr[-1]])
        random.shuffle(l) 
        item = l[0] 
    except:
        item = None
        faulty_id.append(i)
        print("hi - error")
    comp_pairs_lst_1[i] = comp_pairs_lst[i]['item_id_1'].encode('ascii','ignore')
    comp_pairs_lst2[i] = item.encode('ascii','ignore')

print("len of faulty list" , len(faulty_id))

d = {'item_id_1':comp_pairs_lst_1,'item_id_2':comp_pairs_lst2}
df = pd.DataFrame(d)


# comp_pairs.write\
#         .format("com.databricks.spark.csv")\
#         .mode("overwrite")\
#         .option("delimiter", "\01")\
#         .save('amazon/Clothing_Shoes_and_Jewelry/' +"comp_pairs/")
# neg_pairs.write\
#         .format("com.databricks.spark.csv")\
#         .mode("overwrite")\
#         .option("delimiter", "\01")\
#         .save('amazon/Clothing_Shoes_and_Jewelry/' +"neg_pairs/")

#Read GUSE Embeddings
mySchema = StructType([ StructField("item_id", StringType(), True)\
                       ,StructField("ev", StringType(), True)])
embeddings =  spark.read.option("sep","\01")\
                    .schema(mySchema)\
                    .csv('/user/m0m02d5/amazon/Clothing_Shoes_and_Jewelry/text/guse/meta_Clothing_Shoes_and_Jewelry.csv')

def decode_bytes_to_ascii (catalog_item_id):
    res = catalog_item_id.decode("utf-8") 
    import re
    #catalog_item_id.encode('ascii','ignore')
    #res = res.replace('/'b', '')
    res = re.sub('[b\']', '', res)
    return res
decode_bytes_to_ascii_udf = udf(decode_bytes_to_ascii, StringType())
embeddings = embeddings.withColumn("item_id",\
                                             decode_bytes_to_ascii_udf(embeddings.item_id))


comp_pairs = comp_pairs.join(embeddings, embeddings.item_id == comp_pairs.item_id_1)\
                      .select(comp_pairs.item_id_1, comp_pairs.item_id_2, embeddings.ev)\
                      .withColumnRenamed('ev', 'ev1')
comp_pairs = comp_pairs.join(embeddings, embeddings.item_id == comp_pairs.item_id_2)\
                      .select(comp_pairs.item_id_1, comp_pairs.item_id_2, comp_pairs.ev1, embeddings.ev)\
                      .withColumnRenamed('ev', 'ev2')

comp_pairs.write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"comp_pairs_wth_ev/")

neg_pairs = neg_pairs.join(embeddings, embeddings.item_id == neg_pairs.item_id_1)\
                      .select(neg_pairs.item_id_1, neg_pairs.item_id_2, embeddings.ev)\
                      .withColumnRenamed('ev', 'ev1')
neg_pairs = neg_pairs.join(embeddings, embeddings.item_id == neg_pairs.item_id_2)\
                      .select(neg_pairs.item_id_1, neg_pairs.item_id_2, neg_pairs.ev1, embeddings.ev)\
                      .withColumnRenamed('ev', 'ev2')

neg_pairs.write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"neg_pairs_wth_ev/")

sim_pairs = spark.createDataFrame(df)
sim_pairs = sim_pairs.join(embeddings, embeddings.item_id == sim_pairs.item_id_1)\
                      .select(sim_pairs.item_id_1, sim_pairs.item_id_2, embeddings.ev)\
                      .withColumnRenamed('ev', 'ev1')
sim_pairs = sim_pairs.join(embeddings, embeddings.item_id == sim_pairs.item_id_2)\
                      .select(sim_pairs.item_id_1, sim_pairs.item_id_2, sim_pairs.ev1, embeddings.ev)\
                      .withColumnRenamed('ev', 'ev2')
sim_pairs.write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"sim_pairs_wth_ev/")

def calculate_distance(ev1, ev2):
    import numpy as np
    ev1 = np.array(eval(ev1))
    ev2 = np.array(eval(ev2))
    d = float(np.linalg.norm(ev1 - ev2))
    return d
udf_calculate_distance = udf(calculate_distance, DoubleType())

comp_pairs = comp_pairs.withColumn("eu",\
                                             udf_calculate_distance(comp_pairs.ev1, comp_pairs.ev2))
cstats = comp_pairs.select('eu').describe()
print("Complementary Statistics")
cstats.show()

neg_pairs = neg_pairs.withColumn("eu",\
                                             udf_calculate_distance(neg_pairs.ev1, neg_pairs.ev2))
nstats = neg_pairs.select('eu').describe()
print("Negative Pair Statistics")
nstats.show()


sim_pairs = sim_pairs.withColumn("eu",\
                                             udf_calculate_distance(sim_pairs.ev1, sim_pairs.ev2))
sstats = sim_pairs.select('eu').describe()
print("Similar Pair Statistics")
sstats.show()

# Deleting to increase available heap space as we need more space for next collect statements
try:
    del comp_pairs_lst
    del meta_df_categories
    del embeddings
except:
    print("del error")
    
#Collect into array
# eu_comp_pairs = comp_pairs.select('eu').collect()
# eu_neg_pairs = neg_pairs.select('eu').collect()
# eu_sim_pairs = sim_pairs.select('eu').collect()

comp_pairs.select('eu').write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"eu_comp_pairs/")

neg_pairs.select('eu').write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"eu_neg_pairs/")

sim_pairs.select('eu').write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"eu_sim_pairs/")


# eu_comp_pairs = [pair['eu'] for pair in eu_comp_pairs]
# eu_neg_pairs = [pair['eu'] for pair in eu_neg_pairs]
# eu_sim_pairs = [pair['eu'] for pair in eu_sim_pairs]

# f = open("/home/m0m02d5/data/amazon/Clothing_Shoes_and_Jewelry/eu_comp_pairs.pkl","wb")
# pickle.dump(eu_comp_pairs,f)
# f.close()

# f = open("/home/m0m02d5/data/amazon/Clothing_Shoes_and_Jewelry/eu_neg_pairs.pkl","wb")
# pickle.dump(eu_neg_pairs,f)
# f.close()

# f = open("/home/m0m02d5/data/amazon/Clothing_Shoes_and_Jewelry/eu_sim_pairs.pkl","wb")
# pickle.dump(eu_sim_pairs,f)
# f.close()

comp_sim = comp_pairs.join(sim_pairs, comp_pairs.item_id_1 == sim_pairs.item_id_1)\
           .select(comp_pairs.item_id_1.alias('i'),\
                  comp_pairs.item_id_2.alias('c'),\
                  sim_pairs.item_id_2.alias('s')) 

comp_sim_neg = comp_sim.join(neg_pairs, comp_sim.i == neg_pairs.item_id_1)\
           .select(comp_sim.i, comp_sim.c, comp_sim.s,\
                   neg_pairs.item_id_2.alias('n')) 

print('Count comp_sim_neg', comp_sim_neg.count())
if MODE == "test":
    comp_sim_neg = comp_sim_neg.dropDuplicates(['i','c'])
    
comp_sim_neg.select('i','c','s','n')\
        .distinct().orderBy(rand())\
        .write\
        .format("com.databricks.spark.csv")\
        .mode("overwrite")\
        .option("delimiter", "\01")\
        .save(path +"comp_sim_neg_ids")

# ax = sns.distplot(eu_comp_pairs, kde=True, label='complementary', color="r")
# fig = ax.get_figure()
# sns.distplot(eu_neg_pairs, kde=True, label='neg', color="g")
# sns.distplot(eu_sim_pairs, kde=True, label='neg', color="b")
# fig.legend(labels=['Complementary Distance','Non-complementary Distance', 'Similar Distance'])
# path = '/home/m0m02d5/experiments/ctl_v2/v2/baseline_1/figures'
# name = path + "/dist_distribution_clothing_amazon.pdf"
# fig.savefig(name)

