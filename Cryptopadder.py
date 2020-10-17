#This program is used to encrypt and decrypt messages by using a one-time pad approach.
#It's an algorithm with perfect forward secrecy and is the only unbreakable one, if used properly, yet simple to maintain.
#Note that the program is not responsible for key exchange... yet
#For more information visit this link  https://en.wikipedia.org/wiki/One-time_pad


from secrets import randbelow

tempKey = ""
tempClear = ""
tempCipher = ""
passList = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ~!@#$%^&*()_+':;?/.,<>№-=|`[]}{"
listLen = len(passList)

print("Enter one of the 4 options:")
print("\n\n  1 Generate keys")
print("\n  2 Encrypt message")
print("\n  3 Decrypt message")
print("\n  4 Quit")

def key_gen(num=30, lenght=2000, fileName="keys"):
	global listLen
	global passList
	key = ""
	fileName = fileName + ".otp"
	keyFile = open(fileName, mode = "a")
	for i in range(1, num):
		key = ""
		for j in range(1, lenght):
			key = key + passList[randbelow(listLen - 1)]
		if i == 1:
			keyFile.write(key)
		else:
			keyFile.append(f"\n{key}")
	keyFile.close()

def key_read(fileName="keys", lenght=2000):
	key = ""
	fileName = fileName + ".otp"
	keyFile = open(fileName, mode = "+")
	key = keyFile.read(-lenght)
	keyFile.seek(0)
	newF = keyFile.read()
	keyFile.write(newF[(-lenght-1):])
	keyFile.close()
	return key

def assign_value(element):
	global listLen
	global passList
	for i in range(0, listLen - 1):
		if passList[i] == element:
			result = i
	return result

def encrypt_text(text, key):
	global listLen
	global passList
	result = ""
	for i in range(0, len(text)):
		result = result + passList[(assign_value(text[i]) + assign_value(key[i])) % listLen]
	return result

def decrypt_text(text, key):
	global listLen
	global passList
	result = ""
	for i in range(0, len(text)):
		result = result + passList[(assign_value(text[i]) - assign_value(key[i])) % listLen]
	return result

def error_warning(param):
	if param == "1":
		print("This symbol is not allowed, please choose those in the list")

def Main():
	while True:
		pass

Main()