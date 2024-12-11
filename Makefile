me:
	./manage.py dedupe ${PWD}/data/files_me -p 16 --reset --threshold 0.5 -a cosine -e dnn
	./manage.py dedupe ${PWD}/data/files_me -p 16 --reset --threshold 0.5 -a cosine -e retinaface

ch:
	./manage.py dedupe ${PWD}/data/files_ch -p 16 --reset --threshold 0.5 -a cosine -e dnn
	./manage.py dedupe ${PWD}/data/files_ch -p 16 --reset --threshold 0.5 -a cosine -e retinaface


10:
	./manage.py dedupe ${PWD}/data/files_10 -p 16 --reset --threshold 0.5 -a cosine -e dnn
	./manage.py dedupe ${PWD}/data/files_10 -p 16 --reset --threshold 0.5 -a cosine -e retinaface

1000:
	./manage.py dedupe ${PWD}/data/files_1000 -p 16 --reset --threshold 0.5 -a cosine -e dnn
	./manage.py dedupe ${PWD}/data/files_1000 -p 16 --reset --threshold 0.5 -a cosine -e retinaface

10000:
	./manage.py dedupe ${PWD}/data/files_10000 -p 16 --reset --threshold 0.5 -a cosine -e dnn
	./manage.py dedupe ${PWD}/data/files_10000 -p 16 --reset --threshold 0.5 -a cosine -e retinaface
