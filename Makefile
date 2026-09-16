# Makefile for Media Cataloger (AI Engine)
.PHONY: help api cataloger scan test verify info db db-status db-backup db-migrate up down build logs

ENV ?= data/config/.env

help:
	@python manage.py help

api:
	@python manage.py api -e $(ENV)

cataloger:
	@python manage.py cataloger -e $(ENV)

scan:
	@python manage.py scan -e $(ENV)

test:
	@python manage.py test

verify:
	@python manage.py verify -e $(ENV)

info:
	@python manage.py info -e $(ENV)

db:
	@python manage.py db:status -e $(ENV)

db-status:
	@python manage.py db:status -e $(ENV)

db-backup:
	@python manage.py db:backup -e $(ENV)

db-migrate:
	@python manage.py db:migrate -e $(ENV)

up:
	@python manage.py up -e $(ENV)

down:
	@python manage.py down

build:
	@python manage.py build

logs:
	@python manage.py logs
