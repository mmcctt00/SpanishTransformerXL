
imputfile="eswiki/wiki_00"
ouputfile1= "eswiki/train.csv"
ouputfile2= "eswiki/val.csv"
count=1
with open(imputfile, 'r')  as file:
	for line in file:
		if line[0]=='<' or len(line)<20 :
			continue
		else:
			count +=1
			print(count)
			line=line.strip()
			if count%10==0:
				with open(ouputfile2, 'a') as file3:
					file3.write('1	'+line+'\n')
			else:
				with open(ouputfile1, 'a') as file3:
					file3.write('1	'+line+'\n')
