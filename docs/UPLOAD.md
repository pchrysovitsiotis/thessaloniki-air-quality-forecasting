# Publishing this folder on GitHub

Suggested repository name: **`thessaloniki-air-quality-forecasting`**.

Suggested description: **PM10 forecasting for Thessaloniki with ANN, TCN and XGBoost, structured LLM reports, and an offline demo.**

## Using GitHub Desktop

1. Sign in to GitHub Desktop and choose **File → New repository**. Use the suggested name and a new, empty location outside the thesis folders. Leave automatic README, gitignore and license templates unselected; this package already contains the first two, and the license is your choice.
2. Copy the **contents** of this portfolio folder into that new repository. `README.md`, `.gitignore` and `demo.py` should be at the repository root. Do not copy the surrounding thesis workspace. Enable hidden-file display if needed so the dotfiles are included.
3. Run `python demo.py --check` and `python demo.py`. Open the resulting HTML and verify the example.
4. Review the **Changes** list. It should contain source, documentation, configs, examples, small result CSVs and the README figure. The generated `demo_output/` folder is ignored, as are `.env`, data, model files and caches. Do not force-add ignored files.
5. Commit with a message such as **Add forecasting pipeline and offline demo**.
6. Click **Publish repository**. Choose your account and the repository name. Clear **Keep this code private** if you want a public portfolio, then publish.
7. On GitHub, read the rendered README and check that its figure and internal links work. Add topics such as `air-quality`, `time-series`, `pytorch`, `xgboost`, `tcn` and `llm` if useful, and pin the repository on your profile.

If you upload through the website instead, `.gitignore` does not filter a manual file selection. Select only the package files, excluding any `demo_output/`, `.venv/`, `data/`, `outputs/` or other generated folders. GitHub Desktop is the simpler route for keeping those exclusions consistent.

## License and data

No software license has been selected automatically. Before granting reuse rights, check ownership and any university requirements, then add the license you choose. Source-data rights and dependency licenses remain separate. Do not describe the repository as MIT-licensed unless an appropriate MIT license has actually been added.

The complete thesis document and data archive are not needed for this portfolio. They can be published separately later if you decide to do so and have the necessary rights. This prepared folder has not been uploaded or published automatically.

Reference: [GitHub Desktop: creating your first repository](https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop).
