IMAGE=arcface
HOME=/home/coco
run:
	docker build -t ${IMAGE} .
in:
	docker run -v ${HOME}:/home/app --rm -it ${IMAGE}:latest bash
