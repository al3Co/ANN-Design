
# method to do all ANNs & RNNs and save on a dictionary the results given


# options
# [0]All [1]COMBO [2]CRUZEXT [3]CRUZINT [4]ELEFRONT [5]LATERAL [6]ROTZ
# [1]FlexS vs ShoulderAng [2]FlexS+IMUq vs ShoulderAng [3]IMUq vs ShoulderAng [4]PCA vs Shoulder [5]FlexS vs IMUq [6]PCA vs IMUq

movDict = {'all':0,'combo':1,'cruzext':2,'cruzint':3,'elefront':4,'lateral':5,'rotz':6}
sortMov = {'FlexS vs ShoulderAng':1, 'FlexS+IMUq vs ShoulderAng':2,
            'IMUq vs ShoulderAng':3, 'PCA vs Shoulder':4,
            'FlexS vs IMUq':5, 'PCA vs IMUq':6}
results = {}

def reader():
    for mov in sortMov.keys():
        print(mov)
    # for mov in movDict.values():
    #     print(mov)

if __name__ == "__main__":
    reader()
