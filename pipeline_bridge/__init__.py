"""Pipeline Bridge — wraps the image_pipeline for use by the Command Center app.

Provides clean Python APIs for listing methods, generating images, animating,
and promoting results to vault artifacts. All calls go through the pipeline CLI
(subprocess) so the bridge stays decoupled from the pipeline internals.
"""
