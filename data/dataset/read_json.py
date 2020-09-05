import json
f = open('kinetics_labels.json',encoding='utf-8')
content = f.read() #使用loads（）方法需要先读文件
user_dic = json.loads(content)

print(user_dic)
print(user_dic[169])
