def Normalize(features):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=False)
    features = scaler.fit_transform(features)
    return features

def Split(features,labels,test_size=0.3):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = test_size,random_state=0)
    return x_train, x_test, y_train,y_test

