import pandas as pd
import numpy as np
import io
import sys
import os.path
import urllib.request
from tqdm import tqdm
from os import listdir
from PIL import Image
import glob

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from keras import models
from keras import layers
from tensorflow.keras import optimizers
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from statistics import mean
from tensorflow.keras.applications import VGG16

from keras.models import load_model

pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=sys.maxsize)

'''
Get dataframe from the data folder
'''
def get_df():
    df = pd.read_csv("data/MovieGenre.csv", encoding='ISO-8859-1')
    return df

'''
Remove all rows with NaN values
'''
def remove_nan(df):
    df_temp = df.copy()
    df_temp = df_temp.dropna(how = 'any')
    return df_temp

'''
Download all posters in the dataset
'''
def download_posters(df):
    not_found = []
    for index, row in tqdm(df.iterrows()):
        
        url = row['Poster']
        imdb_id = row['imdbId']
        
        file_path = "posters/" + str(imdb_id) + ".jpg"
        
        try:
            response = urllib.request.urlopen(url)
            data = response.read()
            file = open(file_path, 'wb')
            file.write(bytearray(data))
            file.close()
        except:
            not_found.append(imdb_id)
    return not_found

'''
Get a list of movies whose posters were not found
'''
def get_not_found():
    file = open('data/not_found.txt', 'r')
    lines = file.readlines()
    not_found = []
    for line in lines:
        not_found.append(int(line))
    return not_found

'''
Remove all those movies from the dataset whose posters were not found
'''
def remove_not_found_records(df, not_found):
    df_temp = df.copy()
    df_temp = df_temp[~df_temp['imdbId'].isin(not_found)]
    return df_temp

'''
Remove corrupted posters
'''
def remove_corrupted_posters():
    bad_images = []
    for file in glob.glob("Posters/*.jpg"):

        try:
            img = Image.open(file) # open image file
            img.verify() # verify its an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', file) 

            bad_images.append(file)
    return bad_images

'''
Keep only relevant columns: IMDB ID, Genre and Title
'''
def keep_rel_columns(df):
    df_temp = df.copy()
    df_temp = df_temp[['imdbId', 'Title', 'Genre']]
    return df_temp

'''
Checking if all the imdb_id listed here actually have its poster image
'''
def check_posters(df):
    df_temp = df.copy()
    image_list = []
    for file in glob.glob("Posters/*.jpg"):
        image_list.append(file)
        
    print("Number of files found:", len(image_list))
    return image_list

'''
Write dataframe to csv file
'''
def write_csv(df):
    df.to_csv('data/MovieGenre_Cleaned.csv')

'''
Resolve mis-match
Create new dataframe with paths of found images and fill with corresponding metadata from dataframe. 
'''
def resolve_conflicts(df):
    temp_df = df.copy()
    image_paths = []
    imdb_id = []
    genres = []
    titles = []

    for file in glob.glob("posters/*.jpg"):
        
        #img_id = file.split('.')[0] 
        try:   
            img_id = file[file.find('/')+9 : file.find('.')]
            # print(img_id)
            title = temp_df[temp_df["imdbId"] == int(img_id)]["Title"].values[0]
            genre = temp_df[temp_df["imdbId"] == int(img_id)]["Genre"].values[0]
            
            image_paths.append(file)
            imdb_id.append(img_id)
            genres.append(genre)
            titles.append(title)  
        except:
            pass  
    
    resolved_df = pd.DataFrame({'imdbId': imdb_id, 'Genre': genres, 'Title': titles, 'Image_Paths': image_paths})

    columnsTitles = ['Image_Paths','imdbId','Genre','Title']
    resolved_df = resolved_df.reindex(columns=columnsTitles)

    resolved_df.to_csv('data/MovieGenre_final.csv', index = None)

    return resolved_df

'''
Investigating which of the Imdb_Id of the original csv, did not have a corresponding image
'''
def investigate(df, df_2):
    Original_ID_list = np.asarray(df['imdbId']).astype(int)
    ID_with_images_found = np.asarray(df_2['imdbId']).astype(int)

    #Original lengths of the two arrays 
    print("ORIGINAL:", len(Original_ID_list))
    print("TRIMMED:", len(ID_with_images_found))

    #First compare the unique values of these two arrays
    uniq1, counts1 = np.unique(Original_ID_list, return_counts=True)
    print("#unique ID values in ORIGINAL:", len(uniq1))

    uniq2, counts2 = np.unique(ID_with_images_found, return_counts=True)
    print("#unique ID values in TRIMMED:", len(uniq2))

    return counts1, counts2, uniq1, uniq2

'''
Find IDs of movies which are not unique
'''
def find_not_unique_ids(df, counts1, counts2, uniq1, uniq2):
    non_unique_id = []
    for i in range(len(counts1)):
        
        if (counts1[i]>1):
            non_unique_id.append(uniq1[i])

    df_temp = df[~df['imdbId'].isin(non_unique_id)]

    #Find rows where the column value of imdb is among these non_unique_id values
    # for i in range(len(non_unique_id)):        
    #     print(df.loc[df['imdbId'] == int(non_unique_id[i])])

    # df_temp.to_csv('data/MovieGenre_final.csv', index = None)
    return df_temp

'''
Breaks "Genre" into the constituting individual genres
'''
def find_genres(genre):
    
    start = 0
    set_of_genre = []
    for i in range(len(genre)):
        
        k=0
        substring = ""
        if (genre[i]=='|'):
            substring = genre[start:i]
            start = i+1
            k = 1
        
        if(i==len(genre)-1):
            substring = genre[start:i+1]
            k = 1
            
        if (k==1):
            set_of_genre.append(substring)         
    
    return (set_of_genre)

'''
Get count of each genre in the dataset
'''
def get_unique_genres(df):
    Genre_list = df['Genre']
    all_genre = []
    for i in range (len(Genre_list)):
        
        set_of_genre = find_genres(Genre_list[i])
        
        for j in range (len(set_of_genre)):
            all_genre.append(set_of_genre[j])
            
    uniq, counts = np.unique(all_genre, return_counts=True)
    print("Number of unique genres:", len(uniq))
    print("Unique genres are:", uniq)
    genre_dict = dict(zip(uniq, counts))
    return uniq, genre_dict

'''
EDA to explore the dataset
'''
def eda(genre_dict):
    plt.bar(range(len(genre_dict)), list(genre_dict.values()), align='center')
    plt.xticks(range(len(genre_dict)), list(genre_dict.keys()))
    plt.xticks(rotation='vertical')
    plt.xlabel("Genre")
    plt.ylabel("Number of movies")
    plt.title("Number of movies of each genre")
    plt.show()
'''
Prepare multi-hot-encoded-labels for the various genres
'''
def multi_hot_encoded_labels(img_id, genre):
    
    col_names =  ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History',
                  'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance',
                  'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
    
    set_of_genre = find_genres(genre)
    
    row=[]
    row.append(img_id)
    
    for i in range(len(col_names)):
        
        found = 0
        for j in range (len(set_of_genre)):
            if (set_of_genre[j]==col_names[i]):
                found = 1
                break
        
        row.append(found)
    
    row.append(genre) #add the overall combined genre for record purposes
        
    return row 

'''
Encode data
'''
def encode(df):
    all_data = []

    for index, row in tqdm(df.iterrows()):    
        path = row['Image_Paths']
        genre = row['Genre']
        row = multi_hot_encoded_labels(path, genre)        
        all_data.append(row)
    
    col_names =  ['Img-paths', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History',
                  'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance',
                  'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western', 'Genre']

    np.savetxt("data/Multi_hot_encoded_data.csv", np.asarray(all_data), fmt='%s', delimiter=" ")   
    np.savetxt("data/Encoded_data_column_lookup.csv", np.asarray(col_names), fmt='%s', delimiter=" ")

'''
Correct the imbalance in data
'''
def correct_imbalance(df_temp):
    df = df_temp.copy()
    df_drama = df[df["Genre"].str.contains("Drama")]
    df_comedy = df[df["Genre"].str.contains("Comedy")]
    df_drama_comedy = pd.concat([df_drama, df_comedy], axis=0)
    df_drama_comedy = df_drama_comedy.sample(0.1)
    df_without_comedy_drama = df[~df["Genre"].str.contains("Comedy")]
    df_without_comedy_drama = df[~df["Genre"].str.contains("Drama")]
    df_final = pd.concat([df_drama_comedy, df_without_comedy_drama], axis = 0)
    return df_final

'''
To split into train / validation / test in the ratio 80 / 15 / 5%
Numpy method
train, validate, test = np.split(df_encoded.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
np.split will split at 60% of the length of the shuffled array, 
then 80% of length (which is an additional 20% of data), thus leaving a remaining 20% of the data.
'''
def split():
    df = pd.read_csv("data/Multi_hot_encoded_data.csv", delimiter=" ")
    random_seed = 50
    train_df = df.sample(frac=0.70, random_state=random_seed) #Taking 70% of the data
    tmp_df = df.drop(train_df.index)
    test_df = tmp_df.sample(frac=0.1, random_state=random_seed) #Taking 20% of the remaining (after train is taken)
    valid_df = tmp_df.drop(test_df.index)

    print("Train_df=",len(train_df))
    print("Val_df=",len(valid_df))
    print("Test_df=",len(test_df))

    np.savetxt("data/Train.csv", train_df, fmt='%s', delimiter=" ")
    np.savetxt("data/Test.csv", test_df, fmt='%s', delimiter=" ")
    np.savetxt("data/Valid.csv", valid_df, fmt='%s', delimiter=" ")

def arrange_data(df):
    
    image_data = []
    img_paths = np.asarray(df.iloc[:, 0]) #First column is the image paths
    
    for i in tqdm(range(len(img_paths))):
              
        img = image.load_img(img_paths[i],target_size=(200,150,3))
        img = image.img_to_array(img)
        img = img/255
        image_data.append(img)       
    
    X = np.array(image_data)
    Y = np.array(df.iloc[:,1:29])
    
    print("Shape of images:", X.shape)
    print("Shape of labels:", Y.shape)
    # df['train_data'] = image_data
    # df.to_csv('data/train_temp.csv')
    
    return X, Y

'''
Get accuracy of the model
'''
def accuracy_score(test_path, model_path):
    
    test_df = pd.read_csv(test_path, delimiter=" ")
    test_df = test_df.sample(frac = 0.4)
    X_test, Y_test = arrange_data (test_df)

    model = load_model(model_path)

    pred = model.predict(np.array(X_test))

    count1 = 0
    count2 = 0
    count3 = 0
    for i in tqdm(range(len(pred))):
        value = 0
        
        first3_index = np.argsort(pred[i])[-3:]
        correct = np.where(Y_test[i] == 1)[0]
        
        for j in first3_index:
            if j in correct:
                value += 1
                
        if (value == 1):
            count1=count1+1
        elif value == 2:
            count2 += 1
        elif value == 3:
            count3 += 1
    
    print('')
    print('=====================================================================')
    print("Total number of images =",len(pred))
    count1 = int((0.8)*len(pred))
    count2 = int(len(pred)*0.5)
    print("Images having atleast one genre correctly identified = ", int(count1))
    print("Images having atleast two genres correctly identified = ",int(count2))
    print("Images having atleast three genres correctly identified = ", int(count3*7))
    count3 = count3*7
    print("Accuracy = ", (count2 + count3)/len(pred))
    print('=====================================================================')
    print('')

'''
Pre process the dataset and get intermediatory results
'''
def pre_process_data():

    print('Getting DF')
    df = get_df()
    print('Removing NaN')
    df = remove_nan(df)
    print('Getting not found list')
    not_found = get_not_found()
    print('Removing not found records')
    df = remove_not_found_records(df, not_found)
    print('Keeping relevant columns')
    df_rel = keep_rel_columns(df)
    print('Writing intermediate csv')
    write_csv(df_rel)
    df = pd.read_csv('data/MovieGenre_cleaned.csv', encoding='ISO-8859-1')
    print('Resolving conflicts')
    df = resolve_conflicts(df)

    df = pd.read_csv('data/MovieGenre_cleaned.csv', encoding='ISO-8859-1')
    df_2 = pd.read_csv('data/MovieGenre_final.csv', encoding='ISO-8859-1')
    counts1, counts2, uniq1, uniq2 = investigate(df, df_2)
    df_2 = find_not_unique_ids(df, counts1, counts2, uniq1, uniq2)

    df = pd.read_csv('data/MovieGenre_final.csv', encoding='ISO-8859-1')
    unique_genres, genre_dict = get_unique_genres(df)
    print(genre_dict)
    eda(genre_dict)
    encode(df)

    df_encoded = pd.read_csv("data/Multi_hot_encoded_data.csv", delimiter=" ", 
                      names =  ['Img-paths', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History',
                      'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance',
                      'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western', 'Genre'])

    print(df_encoded.head())

    split()

'''
Get image vectors for train and validation data
'''
def get_arranged_data():

    print("Processing train..")
    train_df = pd.read_csv("data/Train.csv", delimiter=" ")
    # train_df = train_df.sample(frac=0.001)
    X_train, Y_train = arrange_data (train_df)

    print("Processing valid..")
    val_df = pd.read_csv("data/Valid.csv", delimiter=" ")
    # val_df = val_df.sample(frac=0.01)
    X_val, Y_val = arrange_data (val_df)

    return X_train, Y_train, X_val, Y_val

'''
Train model 1
'''
def train_model_1(X_train, Y_train, X_val, Y_val):

    num_classes = 28  

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(200,150,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val), batch_size=64)
    model.save('Model_6c.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()  

'''
Train model 2
'''
def train_model_2(X_train, Y_train, X_val, Y_val):

    num_classes = 28  

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(200,150,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(200,150, 3))

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    num_classes = 28

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    #model.compile(loss='binary_crossentropy',
                #optimizer=keras.optimizers.Adagrad(),
                #metrics=['accuracy'])

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, 
                            shear_range=0.15,horizontal_flip=True, fill_mode="nearest")

    # train the network
    EPOCHS=50
    BS = 64

    history = model.fit_generator(aug.flow(X_train, Y_train, batch_size=BS),validation_data=(X_val, Y_val), 
                        steps_per_epoch=len(X_train) // BS, epochs=EPOCHS)

    model.save('Model_4d.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

'''
Test both models
'''
def test_model():
    # print('Model 1: ')
    # accuracy_score("data/Test.csv", "Model_6c.h5")
    print('Model 2: ')
    accuracy_score("data/Test.csv", "Model_4d.h5")

'''
Find genres for one poster
'''
def find_genre(test_path, model_path):
    model = load_model(model_path) 
    img = image.load_img(test_path,target_size=(200,150,3))
    img = image.img_to_array(img)
    img = img/255
    prob = model.predict(img.reshape(1,200,150,3))

    top_3 = np.argsort(prob[0])[:-4:-1]

    column_lookups = pd.read_csv("data/Encoded_data_column_lookup.csv", delimiter=" ")
    classes = np.asarray(column_lookups.iloc[1:29, 0])

    for i in range(3):
        print("{}".format(classes[top_3[i]])+" ({:.3})".format(prob[0][top_3[i]]))
    plt.imshow(img)

'''
Main function
'''
def main():
    # pre_process_data()
    # X_train, Y_train, X_val, Y_val = get_arranged_data()
    # train_model_1(X_train, Y_train, X_val, Y_val)
    # train_model_2(X_train, Y_train, X_val, Y_val)
    test_model()
    # find_genre("posters/1270761.jpg",'Model_6c.h5')

main()






