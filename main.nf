/*
 * AqSolDB aqueous-solubility prediction workflow.
 *
 * Nextflow DSL2 orchestrates inference using the previously
 * trained XGBoost model.
 *
 * The workflow does not train models or evaluate the test set.
 */

nextflow.enable.dsl = 2

// Required command-line parameters.
params.input_csv = null
params.model_path = null


process PREDICT_LOGS {

    tag "molecular-solubility"

    // Docker image containing our Python application.
    container 'aqsoldb-solubility:dev'

    // Publish results outside Nextflow's temporary work directory.
    publishDir 'pipeline_outputs/nextflow',
        mode: 'copy'

    input:
    path molecules_csv
    path trained_model

    output:
    path 'predictions.csv'

    script:
    """
    python -m scripts.predict --input "${molecules_csv}" --model "${trained_model}" --output predictions.csv
    """
}


workflow {

    // Both inputs must be provided when launching the workflow.
    if (!params.input_csv || !params.model_path) {
        error "Provide --input_csv and --model_path."
    }

    // Verify inputs exist before launching the process.
    molecules = file(params.input_csv, checkIfExists: true)
    model = file(params.model_path, checkIfExists: true)

    PREDICT_LOGS(molecules, model)
}
