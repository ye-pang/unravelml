1. Instantiate these containers, note the port mappings

* `docker run --runtime=nvidia -d -v ~/uml:/opt/notebooks -p 30008:8888 -p 30009:6006 -p 30010:5555 ufoym/deepo:all-jupyter-py36-cu100 /bin/bash -c "jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/opt/notebooks"`

* `docker run -p 6379:6379 --name some-redis -d redis`

2. Instantiate the worker 
Command line into the `demo/celery/tasks/` folder and run the following command:

`celery -A unravelML worker -n worker1@%n`

3. Instantiate the flower monitoring server on port 5555 from the same path above:

`celery -A unravelML flower --port=5555`

3. Instantiate the Dash server

