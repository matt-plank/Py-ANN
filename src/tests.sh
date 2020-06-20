HEADER="=========="

function print_h() {
  echo
  echo $HEADER $1 $HEADER
  echo
}

if [[ "$1" == "full" ]]
then
  print_h "Testing Installation Process"
  print_h "Installing PyANN with Docker"
  docker image build -t py_ann_test .

  print_h "Running PyANN examples with Docker"
  docker container run py_ann_test
fi

print_h "Testing Types"
python3 -m pytype PyANN/* || exit 1

print_h "Running Unit Tests"
python3 -m unittest PyANN/tests/test_*