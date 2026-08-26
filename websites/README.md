# Website configurations

League files in `../leagues/` contain only fantasy-platform identity. Website files contain repository, deployment, and output details and reference a league by its filename slug.

This separation supports:

- a league with no website;
- multiple websites for one league;
- a reusable website implementation deployed separately for different leagues.

Credentials and deployment tokens must remain in environment variables or repository secrets.
