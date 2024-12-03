import subprocess
import threading
import rospy
import os


ROBOT_PACKAGE_NAME: str = "robot_controller"
LAUNCH_CMD = lambda name: ['rosrun', ROBOT_PACKAGE_NAME, name]


class NodeThread:
    def __init__(self, name):
        self._node_name = name

        # We want to run our nodes, so we will start a new process and two threads 
        # to capture their stdout and stderr and pipe it back to this main process
        self._node_process: subprocess.Popen = None
        self._stdout_thread: threading.Thread = None
        self._stderr_thread: threading.Thread = None

    def start(self):
        """
        Try to launch a node, if it doesn't already exist.
        """
        self._node_process, self._stdout_thread, self._stderr_thread = self._launch_node(self._node_process, LAUNCH_CMD(self._node_name))

    def kill(self):
        """
        Stop a node, if it is running.
        """
        self._kill_node(self._node_process, self._stdout_thread, self._stderr_thread)

    def _kill_node(self, process: subprocess.Popen, stdout_thread: threading.Thread, stderr_thread: threading.Thread):
        """
        Stop a node, if it is running.
        """
        if process and process.poll() is None:
            process.terminate()

            # Kill the logging threads
            stderr_thread.join()
            stdout_thread.join()

            process.wait()

            rospy.loginfo("Robot Node has been stopped.")

        else:
            rospy.logerr("Cannot stop Robot Node: not started!")

    def _launch_node(self, process: subprocess.Popen, cmd):
        """
        Try to launch a node, if it doesn't already exist.
        """
        if process is None or process.poll() is not None:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1")
            )

            # We will start two threads that will read the node process's stdout and stderr and log it
            def stream_stdout():
                """Thread function to read and print stdout line by line."""
                for line in process.stdout:
                    rospy.loginfo(line)

            def stream_stderr():
                """Thread function to read and print stderr line by line."""
                for line in process.stderr:
                    rospy.logerr(line)

            _stdout_thread = threading.Thread(target=stream_stdout)
            _stderr_thread = threading.Thread(target=stream_stderr)

            _stdout_thread.start()
            _stderr_thread.start()

            rospy.loginfo(f"Node {self._node_name} has been started.")

            return process, _stdout_thread, _stderr_thread
            
        else:
            rospy.logerr("Cannot start node {self._node_name}: already started!")


