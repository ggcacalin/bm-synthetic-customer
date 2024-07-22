Repo contains all files necessary to run the agent and/or upload it to Docker Hub.

Will only run on Linux due to a number of packages not being supported on Windows.

Dockerfile assumes everything is moved to a working directory named "syncus". Modify the dockerfile if this is not the case.

All paths are written as repo variables and should be added to an .env file together with the secrets.
