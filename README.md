Repo contains all files necessary to run the agent and/or upload it to Docker Hub.

To run the agent, three arguments must be supplied in JSON form: a session_id (used to create or fetch chat memory), a mosaic_id (used to identify the synthetic customer the buyer is trying to talk to) and the input_message the AI is supposed to respond to.

Will only run on Linux due to a number of packages not being supported on Windows.

Dockerfile assumes everything is moved to a working directory named "syncus". Modify the dockerfile if this is not the case.

All paths are written as repo variables and should be added to an .env file together with the secrets.

Sample bot startup (Flask / Gunicorn):
sudo -E env "PATH=$PATH" PYTHONPATH=src gunicorn -w 2 -b 0.0.0.0:80 gpt_4o_agent:agent_app

Sample bot interaction:
curl -X POST -H "Content-Type: application/json" -d '{"session_id": "devtest_session", "mosaic_id": "established_wealth", "input_message": "How much do you spend on it weekly?"}' http://localhost:80/api/syncus
