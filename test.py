# dic = {'one': 'big time', 'two': 'small time', 'three': 'I wannit'}
# ftwo = {k: dic[k] for k in list(dic)[:2]}
# print(ftwo)


import pandas as pd
df = pd.DataFrame(columns=['username', 'password'])
print(df)
creds = {}
count = 0
while count <= 1:
    ch = input('Enter Name: ')
    pw = input('Enter pw: ')
    count += 1

    df.loc['username'] = ch
    df.loc['password'] = pw


print(df)
print(creds)



