# PhishLens — PhiUSIIL Phishing Detection Demo

Angular frontend + Flask inference API + GitHub Actions CI/CD.

## Important limitation
The first version accepts numeric feature vectors from the PhiUSIIL model schema. It does not extract webpage features from a URL. Do not describe this as a live URL scanner. Check the official dataset documentation for label mapping.

## Local setup

1. Put the PhiUSIIL CSV in `data/`.
2. Create and activate a virtual environment, then install `backend/requirements.txt`.
3. Run `python backend/train.py` from the project root.
4. Start Flask with `python backend/app.py`.
5. Create the Angular project in `frontend/` as described in `frontend/README.md`, then copy the supplied Angular source files into it.
6. In a second terminal: `cd frontend && npm install && npm start`.
7. Open the local URL printed by Angular.

## Deployment
GitHub Pages hosts only the Angular static frontend. Deploy Flask separately as a web service (for example Render). Configure the Angular API URL to the deployed HTTPS Flask URL, then commit and push. In GitHub repository Settings → Pages, choose GitHub Actions as the source.

## Research caution
Random-split results can be optimistic. Audit duplicates, target leakage, and domain overlap; report a domain-disjoint or independent evaluation where possible. A high accuracy number is not evidence of real-world reliability by itself.
