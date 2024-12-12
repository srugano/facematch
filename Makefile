me:
	./manage.py dedupe ${PWD}/data/files_me -p 16 --reset --threshold 0.5 

ch:
	./manage.py dedupe ${PWD}/data/files_ch -p 16 --reset --threshold 0.5 

dina:
	./manage.py dedupe ${PWD}/data/files_dina -p 16 --reset --threshold 0.5 

10:
	./manage.py dedupe ${PWD}/data/files_10 -p 16 --reset --threshold 0.5 

1000:
	./manage.py dedupe ${PWD}/data/files_1000 -p 16 --reset --threshold 0.5 


2000:
	./manage.py dedupe ${PWD}/data/files_2000 -p 16 --reset --threshold 0.5 

10000:
	./manage.py dedupe ${PWD}/data/files_10000 -p 16 --reset --threshold 0.5 
