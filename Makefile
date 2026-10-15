# uncompute — optional C kernel for the evaluator, the serializer, and the simulator.
#
# The kernel is a speed optimization only: the pure-Python reference evaluator
# is normative and stays usable without it.
#
#   make kernel        build if out of date (prints the library path)
#   make -B kernel     force a rebuild
#   make clean-kernel  remove the built library
#   make replay-kernel build tools/liveness_profile.py's optional C replay
#                      core (tools/_replay_kernel.c) if out of date
#
# The library path comes from LIB_PATH in uncompute/evalcore/kernel.py, the module
# that loads the library. The build and the loader therefore use the same path.

# The project venv when present (it has uncompute's dependencies), else python3.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CC ?= cc

SRC := uncompute/evalcore/_kernel.c
LIB := $(shell $(PYTHON) -c "from uncompute.evalcore.kernel import LIB_PATH; print(LIB_PATH)" 2>/dev/null)
ifeq ($(strip $(LIB)),)
$(error cannot read LIB_PATH from uncompute/evalcore/kernel.py using PYTHON=$(PYTHON) -- run make from the repository root, or set PYTHON to a working interpreter)
endif

# CPU-feature flags: arm64 gets the SHA3 Keccak instructions and x86-64 gets
# -march=native (which selects _kernel.c's eight-way Keccak draw() branch when
# the host has AVX-512F). Both give results identical to the scalar path. The
# flags are safe because the host that runs the library also builds it. If the
# compiler rejects the flags, the plain compile below runs instead, and
# _kernel.c's #if guard selects the scalar path.
MACHINE := $(shell uname -m)
ifeq ($(MACHINE),$(filter $(MACHINE),arm64 aarch64))
ARCH_FLAGS := -march=armv8.4-a+sha3
else ifeq ($(MACHINE),$(filter $(MACHINE),x86_64 amd64))
ARCH_FLAGS := -march=native -mprefer-vector-width=512
else
ARCH_FLAGS :=
endif

CFLAGS ?= -O3 -shared -fPIC

.PHONY: kernel clean-kernel replay-kernel

kernel: $(LIB)
	@echo $(LIB)

# The compiler writes to a per-process temporary name in the same directory.
# `mv` then moves the file into place atomically, so concurrent builds never
# expose a half-written library to a process that loads it.
$(LIB): $(SRC)
	@command -v $(CC) >/dev/null 2>&1 || { \
	  echo "no C compiler found ($(CC)); cannot build the C kernel" >&2; exit 1; }
	@tmp="$(LIB).tmp$$$$"; \
	if $(CC) $(CFLAGS) $(ARCH_FLAGS) -o "$$tmp" $(SRC) 2>/dev/null \
	   || $(CC) $(CFLAGS) -o "$$tmp" $(SRC); then \
	  mv -f "$$tmp" "$(LIB)"; \
	else \
	  rm -f "$$tmp"; \
	  echo "kernel compile failed: $(CC) $(CFLAGS) -o $(LIB) $(SRC)" >&2; \
	  exit 1; \
	fi

clean-kernel:
	@rm -f $(LIB) $(LIB).tmp*

# The replay library's path is read from tools/liveness_profile.py
# (REPLAY_LIB_PATH), which also loads it; it needs that module's imports
# (numpy, click), so PYTHON must be the project interpreter.
REPLAY_SRC := tools/_replay_kernel.c

replay-kernel:
	@lib=$$($(PYTHON) -c "import sys; sys.path.insert(0, 'tools'); \
from liveness_profile import REPLAY_LIB_PATH; print(REPLAY_LIB_PATH)") \
	  || { echo "cannot read REPLAY_LIB_PATH (set PYTHON to the project interpreter)" >&2; exit 1; }; \
	if [ ! -f "$$lib" ] || [ $(REPLAY_SRC) -nt "$$lib" ]; then \
	  tmp="$$lib.tmp$$$$"; \
	  $(CC) -O3 -shared -fPIC -o "$$tmp" $(REPLAY_SRC) && mv -f "$$tmp" "$$lib" \
	    || { rm -f "$$tmp"; echo "replay kernel compile failed" >&2; exit 1; }; \
	fi; \
	echo "$$lib"
