me:
	./manage.py dedupe ${PWD}/data/files_me -p 16 --reset --threshold 0.5 -e dnn
	./manage.py dedupe ${PWD}/data/files_me -p 16 --reset --threshold 0.5 -e retinaface

ch:
	./manage.py dedupe ${PWD}/data/files_ch -p 16 --reset --threshold 0.5 -e dnn
	./manage.py dedupe ${PWD}/data/files_ch -p 16 --reset --threshold 0.5 -e retinaface

dina:
	./manage.py dedupe ${PWD}/data/files_dina -p 16 --reset --threshold 0.5 -e dnn
	./manage.py dedupe ${PWD}/data/files_dina -p 16 --reset --threshold 0.5 -e retinaface

10:
	./manage.py dedupe ${PWD}/data/files_10 -p 16 --reset --threshold 0.5 -e dnn
	./manage.py dedupe ${PWD}/data/files_10 -p 16 --reset --threshold 0.5 -e retinaface

1000:
	./manage.py dedupe ${PWD}/data/files_1000 -p 16 --reset --threshold 0.5 -e dnn
	./manage.py dedupe ${PWD}/data/files_1000 -p 16 --reset --threshold 0.5 -e retinaface

10000:
	./manage.py dedupe ${PWD}/data/files_10000 -p 16 --reset --threshold 0.5 -e dnn
	./manage.py dedupe ${PWD}/data/files_10000 -p 16 --reset --threshold 0.5 -e retinaface
