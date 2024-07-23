Repo contains all files necessary to run the agent and/or upload it to Docker Hub.

To run the agent, two arguments must be supplied in order: a session ID (used to create or fetch chat memory) and the message the AI is supposed to respond to.

Will only run on Linux due to a number of packages not being supported on Windows.

Dockerfile assumes everything is moved to a working directory named "syncus". Modify the dockerfile if this is not the case.

All paths are written as repo variables and should be added to an .env file together with the secrets.
