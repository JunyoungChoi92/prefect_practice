from prefect import Flow
from prefect.schedules import CronSchedule
from tasks.train_1 import (
    load_dataset,
    hpo_task,
    train_task
)

with Flow(name ="apartment_tutorial",
          schedule=CronSchedule("* * * * *"),
) as flow:
    train, valid = load_dataset()
    # preprocesser, best_params, best_values = hpo_task(train, valid)
    prep_pipeline, params, metrics = hpo_task(
        train, valid, upstream_tasks=[train, valid]
    )

    model = train_task(
        prep_pipeline,
        train,
        valid,
        params,
        upstream_tasks=[prep_pipeline, params, metrics],
    )

if __name__ == "__main__":
    flow.register(project_name="prefect-tutorial",
                  add_default_labels=False,
                  labels=['train_agent'])

