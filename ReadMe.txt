1. Download and Install CoppeliaSim Edu version
2. To run the simulator with the provided scene, use this command:
	For mac: /Applications/coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 /users/Sadman/Downloads/drive/Intro_AI_Mix/mix_intro_AI.ttt
	For ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 /users/Sadman/Downloads/drive/Intro_AI_Mix/mix_intro_AI.ttt
3. Install ZeroMQ and Cbor:
	python3 -m pip install pyzmq
	python3 -m pip install cbor
4. Now you need to copy the zmqRemoteApi folder to the project directory. 
5. Run the python script:
	python3 exec_environment.py