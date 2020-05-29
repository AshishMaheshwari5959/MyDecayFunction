import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

dataset = pd.read_csv('weight-height.csv')

y = dataset['Weight']
X = dataset[['Height']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = Sequential()

model.add(Dense(units=1 , input_shape=(1,)  ))

earlystop = EarlyStopping(monitor = 'loss', min_delta = 0.001,patience = 3,verbose = 1,restore_best_weights = True)
checkpoint = ModelCheckpoint("weight.h5",monitor="loss",mode="min",save_best_only = True,verbose=1)

callbacks = [earlystop]

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.1) )
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks = callbacks,epochs=10,verbose=1)

def compile(x):
    global c
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=x) )
    hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks = callbacks,epochs=10,verbose=1)
    a = hist.history['loss'][-1]
    b = hist.history['loss'][-2]
    c = abs(a-b)
    return c
c=2
l =[]
s=0
while c > 0.000001 :
    print(c)
    if c >= 10 :
        compile(0.01*c)
    elif 10 > c >= 1:
        compile(0.1*c)
    else :
        compile(0.01*c)
    l.append(model.evaluate(X_train,y_train,verbose=0))
    s=s+1
    if s >4 :
        p = l[-1]
        q = l[-2]
        r = l[-3]
        if p>q and p>r :
            break
        if l[-1] == min(l):
            model.save('weight.h5')
        
model = load_model('weight.h5')
print(model.evaluate(X_train,y_train,verbose=0))