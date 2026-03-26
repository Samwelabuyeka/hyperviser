# AURORA Compute Scheduler Kernel Module Makefile

obj-m += aurora_compute.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -O3 -march=native -mtune=native

# Module version
VERSION := 0.1.0

.PHONY: all clean install uninstall load unload

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.o *.ko *.mod.c *.mod.o *.order *.symvers .*.cmd

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a
	@echo "AURORA kernel module installed"

uninstall:
	rm -f /lib/modules/$(shell uname -r)/extra/aurora_compute.ko
	depmod -a
	@echo "AURORA kernel module uninstalled"

load:
	modprobe aurora_compute || insmod aurora_compute.ko
	@echo "AURORA kernel module loaded"

unload:
	rmmod aurora_compute 2>/dev/null || true
	@echo "AURORA kernel module unloaded"

reload: unload load

# Development targets
debug:
	$(MAKE) -C $(KDIR) M=$(PWD) modules ccflags-y="-DDEBUG -g"

# Check kernel headers
check:
	@if [ ! -d "$(KDIR)" ]; then \
		echo "Error: Kernel headers not found at $(KDIR)"; \
		echo "Install kernel headers: sudo apt-get install linux-headers-$(shell uname -r)"; \
		exit 1; \
	fi
	@echo "Kernel headers found at $(KDIR)"

# Print module info
info:
	@echo "AURORA Compute Scheduler Kernel Module"
	@echo "Version: $(VERSION)"
	@echo "Kernel: $(shell uname -r)"
	@echo "Build dir: $(KDIR)"
