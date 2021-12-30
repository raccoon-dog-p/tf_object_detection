# tf_object_detection
TensorFlow2  object Detection 설치 방법


conda create -n tensorflow pip python=3.9

pip install --ignore-installed --upgrade tensorflow==2.5.0

https://github.com/tensorflow/models 에서 파일 클론 후 원하는 경로에 압축 풀기

 https://github.com/google/protobuf/releases 주소 들어간후 내 os에 맞는 집 파일 다운 후
c드라이브 => 프로그램 파일스에 Google protobut에 압축풀기

Environment Setup
=> 내 컴퓨터 들어가서 속성
설정 => 고급 시스템 설정 =>  환경 변수 => 시스템 환경 변수
=> path 에 새로만들기 => 프로그램 파일즈에 Google protobut / bins 경로 추가

# From within TensorFlow/models/research/
protoc object_detection/protos/*.proto --python_out=. 

pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

명령어 통해서 설치.
(주의사항)
반드시 비주얼 스튜디오 C++ 패키지가 설치되어 있어야 합니다.


# From within TensorFlow/models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .


# From within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py => 코드를 통하여 제대로 테스트 되는지 확인
