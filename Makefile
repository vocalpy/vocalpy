SOURCE_TEST_DATA_TAR=tests/test_data/source/source_test_data.tar.gz
SOURCE_TEST_DATA_URL=https://osf.io/7ru4s/download

GENERATED_TEST_DATA_TAR=tests/test_data/generated/generated_test_data.tar.gz
GENERATED_TEST_DATA_URL=https://osf.io/q76xd/download
GENERATED_TEST_DATA_TOP_LEVEL_DIRS=tests/test_data/generated/configs tests/test_data/generated/prep tests/test_data/generated/results

help:
	@echo 'Makefile for vak                                                           			'
	@echo '                                                                           			'
	@echo 'Usage:                                                                     			'
	@echo '     make test-data-clean-source          remove source test data                        '
	@echo '     make test-data-download-source       download source test data                      '
	@echo '     make test-data-clean-generate        remove generated test data          					'
	@echo '     make test-data-download-generate     download generated test data .tar and expand        	'
	@echo '     make variables              show variables defined for Makefile 					'

variables:
	@echo '     SOURCE_TEST_DATA_TAR                : $(SOURCE_TEST_DATA_TAR)				'
	@echo '     SOURCE_TEST_DATA_URL                : $(SOURCE_TEST_DATA_URL)				'
	@echo '     GENERATED_TEST_DATA_TAR      		: $(GENERATED_TEST_DATA_TAR)				'
	@echo '     GENERATED_TEST_DATA_URL      		: $(GENERATED_TEST_DATA_URL)				'
	@echo '     GENERATED_TEST_DATA_TOP_LEVEL_DIRS	: $(GENERATED_TEST_DATA_TOP_LEVEL_DIRS)		'

test-data-clean-source:
	rm -rfv ./tests/test_data/source/*

test-data-download-source:
	wget -q $(SOURCE_TEST_DATA_URL) -O $(SOURCE_TEST_DATA_TAR)
	tar -xzf $(SOURCE_TEST_DATA_TAR)

test-data-clean-generate :
	rm -rfv ./tests/test_data/generated/*

test-data-download-generate:
	wget -q $(GENERATED_TEST_DATA_URL) -O $(GENERATED_TEST_DATA_TAR)
	tar -xzf $(GENERATED_TEST_DATA_TAR)

.PHONY: help variables test-data-clean-source test-data-download-source test-data-clean-generate test-data-download-generate
