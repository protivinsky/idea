from abc import ABC, abstractmethod

# The goal of this exercise is to implement the Command pattern with all the proper Python syntax and typing.

class CommandInterface(ABC):
    """ The Command interface. """
    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError()


class Switch:
    """ The Invoker class. """
    _closedCommand: CommandInterface
    _openedCommand: CommandInterface

    def __init__(self, closedCommand, openedCommand):
        self._closedCommand = closedCommand
        self._openedCommand = openedCommand

    def close(self) -> None:
        self._closedCommand.execute()

    def open(self) -> None:
        self._openedCommand.execute()


class SwitchableInterface(ABC):
    """ An interface that defines actions that the receiver can perform. """

    @abstractmethod
    def power_on(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def power_off(self) -> None:
        raise NotImplementedError


class Light(SwitchableInterface):
    """ The Receiver class. """

    def power_on(self):
        print("The light is on.")

    def power_off(self):
        print("The light is off.")


class CloseSwitchCommand(CommandInterface):
    """ The Command for turning off the device. """
    _switchable: SwitchableInterface

    def __init__(self, switchable: SwitchableInterface):
        self._switchable = switchable

    def execute(self) -> None:
        self._switchable.power_off()


class OpenSwitchCommand(CommandInterface):
    """ The Command for turning on the device. """
    _switchable: SwitchableInterface

    def __init__(self, switchable: SwitchableInterface):
        self._switchable = switchable

    def execute(self) -> None:
        self._switchable.power_on()



lamp: SwitchableInterface = Light()
switchClose: CommandInterface = CloseSwitchCommand(lamp)
switchOpen: CommandInterface = OpenSwitchCommand(lamp)

switch: Switch = Switch(switchClose, switchOpen)

switch.open()
switch.close()


