#Graham's Number generator


def arrow(num1, num2):   #tetration
	for i in range(1, num2):
		if i == 1:
			result = num1 ** num1
			break
		else:
			result = num1 ** result
	return result

def arrowNum(num1, num2, amount=1):  #determines the amount of arrows to be defined
	if amount == 1:
		result = num1 ** num2
	elif amount == 2:
		result = arrow(num1, num2)
	else:
		amount -= 2
		result = arrow(num1, num2)
		for i in range(1, (num2 - 1)**amount):
			result = arrow(num1, result)
	return result

graham = arrowNum(3, 3, 6)

for g in range(1, 64):
	graham = arrowNum(3, 3, graham)

print(graham)