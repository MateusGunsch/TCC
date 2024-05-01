#!/usr/bin/python3
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.query import Query
from deepface import DeepFace
import numpy as np
import os
import time

NUMBER_SAMPLES = 200
FACE_IMAGE_VECTOR_FIELD='face_image_vector'
modelname = "VGG-Face"

if (modelname == "VGG-Face"):
    IMAGE_VECTOR_DIMENSION=4096
elif (modelname == "Facenet"):
    IMAGE_VECTOR_DIMENSION=128
elif (modelname == "Facenet512"):
    IMAGE_VECTOR_DIMENSION=512
elif (modelname == "OpenFace"):
    IMAGE_VECTOR_DIMENSION=128
elif (modelname == "DeepID"):
    IMAGE_VECTOR_DIMENSION=160
elif (modelname == "ArcFace"):
    IMAGE_VECTOR_DIMENSION=512
elif (modelname == "SFace"):
    IMAGE_VECTOR_DIMENSION=128
elif (modelname == "GhostFaceNet"):
    IMAGE_VECTOR_DIMENSION=512

r = redis.Redis(host='127.0.0.1', port=6379, password='')
r.flushdb()


if os.path.isfile("results_olivetti_deepface.txt"):
    os.remove("results_olivetti_deepface.txt")

f = open("results_olivetti_deepface.txt", "a")
f.write("Model used: " + modelname + "\n")

def store_olivetti_models():
    global r
    for person in range(1, 41):
        person = "s" + str(person)
        for face in [1, 3, 5, 7, 9]:
        #for face in [2, 4, 6, 8, 10]:
        #for face in range(1, 6):
            facepath = 'databases/olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Training face: " + facepath)
            embedding = DeepFace.represent(img_path=facepath, model_name=modelname, enforce_detection=False)[0]["embedding"]
            face_image_vector = np.array(embedding).astype(np.float32).tobytes()
            face_data_values ={ 'person_id':person,
                                'person_path':facepath,
                                  FACE_IMAGE_VECTOR_FIELD:face_image_vector}
            r.hset('face_'+person+'_'+str(face),mapping=face_data_values)


def test_olivetti_models_vect():
    success = 0
    for person in range(1, 41):
        person = "s" + str(person)
        for face in [2, 4, 6, 8, 10]:
        #for face in [1, 3, 5, 7, 9]:
        #for face in range(6, 11):
            facepath = 'databases/olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Testing face: " + facepath)
            found = find_face(facepath)
            if (person == found):
                success = success +1
                print ('*** Face found ***')
                print(person)
                print ('Match:' + found)
            else:
                f.write("Face not found:" + person + "/" + str(face) + "\n")


    print(success/200*100)


def create_hnsw_index (redis_conn,index_name,vector_field_name,number_of_vectors, vector_dimensions=IMAGE_VECTOR_DIMENSION, distance_metric='L2',M=40,EF=200):
    global r
    schema = (VectorField("face_image_vector", "HNSW", {"TYPE": "FLOAT32", "DIM": IMAGE_VECTOR_DIMENSION, "DISTANCE_METRIC": "L2"}),)
    hnsw_index = r.ft().create_index(schema)
    return hnsw_index


def find_face(path):
    global r

    embedding = DeepFace.represent(img_path=path, model_name=modelname, enforce_detection=False)[0]["embedding"]
    face_image_vector = np.array(embedding).astype(np.float32).tobytes()

    q = Query("*=>[KNN 1 @face_image_vector $vec]").return_field("__face_image_vector_score").dialect(2)
    res = r.ft().search(q, query_params={"vec": face_image_vector})

    for face in res.docs:
        ##print ('*** Face found ***')
        ##print(face.id.split("_")[1])
        return face.id.split("_")[1]

start = time.perf_counter()

my_hnsw_index = create_hnsw_index(r,'my_hnsw_index',FACE_IMAGE_VECTOR_FIELD,NUMBER_SAMPLES,IMAGE_VECTOR_DIMENSION,'L2',M=40,EF=200)
store_olivetti_models()
start_test = time.perf_counter()

test_olivetti_models_vect()

end = time.perf_counter()

tempo = end - start
tempo_teste = end - start_test
print(f"Tempo de execucao: {tempo: .5f} segundos")
print(f"Tempo de teste: {tempo_teste: .5f} segundos")