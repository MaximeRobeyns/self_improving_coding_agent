.PHONY: test

PWD := $(shell pwd)

# Credentials are not built into the image, but handed to each container.
# `-e NAME` passes a variable on from your shell without its value appearing
# on the command line; variables that are not set are simply left out.
SANDBOX_SECRETS := \
	-e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e FIREWORKS_AI_API_KEY \
	-e GEMINI_API_KEY -e DEEPSEEK_API_KEY -e VERTEX_PROJECT_ID -e VERTEX_REGION \
	-v ${PWD}/sandbox/GOOGLE_APPLICATION_CREDENTIALS.json:/tmp/GOOGLE_APPLICATION_CREDENTIALS.json:ro \
	-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/GOOGLE_APPLICATION_CREDENTIALS.json

int:  ## Interactive run; uses default shell entrypoint
	@echo 'Once in the container, type:'
	@echo 'python -m agent_code.agent -s -p "<your prompt here>"'
	@echo 'Watch the agent work on localhost:8080'
	docker run --rm -ti \
		-p 8080:8080 \
		$(SANDBOX_SECRETS) \
		-v ${PWD}/base_agent:/home/agent/agent_code:ro \
		-v ${PWD}/results/interactive_output:/home/agent/workdir:rw \
		sica_sandbox

test:  ## Run the unit tests for the agent and the runner
	@pytest base_agent
	@pytest tests

image:  ## Docker image for x86_64
	docker buildx build --build-context base_agent=./base_agent \
		-f sandbox/Dockerfile \
		-t sica_sandbox \
		--build-arg TARGET_ARCH=x86_64 \
		--load sandbox

image-mac:  ## Docker image for apple silicon
	docker buildx build --build-context base_agent=./base_agent \
		-f sandbox/Dockerfile \
		-t sica_sandbox \
		--build-arg TARGET_ARCH=aarch64 \
		--load sandbox

docs:  ## Compile documentation
	python base_agent/utils/documentation.py base_agent > base_agent/DOCUMENTATION.md

meta:  ## Run the meta-agent agent directly for testing (see manual request in __main__.py)
	rm -rf results/meta
	mkdir -p results/meta/test_logs
	cp -r base_agent results/meta/agent_iter
	# Copy an existing archive so that the meta agent has something to work with
	cp -r results/run_1 results/meta/archive
	@echo localhost:8080
	docker run --rm -ti \
		-p 8080:8080 \
		$(SANDBOX_SECRETS) \
		-v ${PWD}/base_agent:/home/agent/meta:ro \
		-v ${PWD}/results/meta/archive:/home/agent/archive:ro \
		-v ${PWD}/results/meta/agent_iter:/home/agent/workdir:rw \
		-v ${PWD}/results/meta/test_logs:/home/agent/meta_logdir:rw \
		sica_sandbox python -m meta improve \
		--workdir /home/agent/workdir \
		--logdir /home/agent/meta_logdir

test_meta_int:  ## Interactivley test the resulting agent from the target above
	docker run --rm -ti \
		-p 8080:8080 \
		$(SANDBOX_SECRETS) \
		-p 8000:8000 \
		-v ${PWD}/results/meta/agent_iter:/home/agent/agent_code:ro \
		-v ${PWD}/results/meta/test_output:/home/agent/workdir:rw \
		sica_sandbox


help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean-sandbox:
	docker rm -f sica_sandbox

deepclean:
	rm -rf .venv base_agent/__pycache__ results
	docker rmi sica_sandbox

clean-build:
	@$(MAKE) clean-sandbox
	@$(MAKE) deepclean
build:
	pip install -r base_agent/requirements.txt
rebuild:
	$(MAKE) clean-build;
	python3 -m venv .venv && source .venv/bin/activate
	$(MAKE) build
	$(MAKE) image-mac	


