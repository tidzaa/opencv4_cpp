## opencv4_cpp
this is an opencv4 cpp project collections
# OpenCV----Classification Demo With ml

#### Features

- 1）support command line, change test image, light pattern and model
- 2）support default model select as SVM;
- 3）support template class programing;
- 4）the whole project is well managed as folder;
- 5）the whole code is well classified as utils;

#### usages

```bash
 ./bin/main -h
```

```
Usage: main [params] image lightPat mode 

        -?, -h, --help, --usage (value:true)
                Print this message

        image
                Image for test
        lightPat
                light pattern for test image
        mode (value:svm)
                machine learning mode, default svm
```

```bash
 ./bin/main path_to_test_image  path_for_light_pattern model_name
```
