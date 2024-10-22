# MLflow

Environnement MLflow avec docker, compatible heroku.

## Process d'installation du projet

```bash
heroku container:login
heroku create --region eu mlflow-luciole
heroku stack:set container -a  mlflow-luciole
heroku container:push web -a mlflow-luciole
heroku container:release web -a mlflow-luciole
```

## Lancer le container en local

```bash
docker run -it -p 8080:8080\
 -v "$(pwd):/mlflow"\
 -e PORT=8080\
 container-name
```

## Lancer un entrainement

```bash
docker run -it -p 8080:8080\
 -v "$(pwd):/mlflow"\
 -e PORT=8080\
 -e AWS_ACCESS_KEY_ID="xxx"\
 -e AWS_SECRET_ACCESS_KEY="xxx"\
 -e AWS_ARTIFACT_S3_URI="s3://xxx-bucket/xxx-artifacts/"\
 -e DATABASE_URL="postgres://xxxx"\
 container-name python train.py
```
