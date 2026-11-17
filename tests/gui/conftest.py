import pytest


@pytest.fixture(autouse=True)
def _drain_qt_workers():
    """Don't let a test's background workers (prefetch, simulation) outlive it.

    Leftover QRunnables on the shared global pool used to crash a later test natively
    while Qt delivered their events to widgets that had already been torn down.
    """
    yield
    from qtpy import QtCore, QtWidgets

    QtCore.QThreadPool.globalInstance().waitForDone(10000)
    if QtWidgets.QApplication.instance() is not None:
        QtWidgets.QApplication.processEvents()
