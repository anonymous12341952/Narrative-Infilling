from typing import List, Tuple, Dict, Optional
import re
import copy

from infill_workflow.utils import strings

class Session:
    """A wrapper for chat/inference dialogues
    
    This provides an interface for managing chat sessions with LLMs.
    This is meant to be used in conjunction with the SessionManager 
    class. 
    """
    def __init__(self, 
        sess_type='chat', 
        system_role: Optional[str]=None,
        hist: List[Dict[str, str]] | List[str]=[],
        override_sys: bool=False,
        supports_sys: bool=True,
    ):
        """
        Args:
            sess_type: a string that is either 'chat' or 'standard'. 
              (Default: 'chat')
        Raises:
            ValueError: argument 'sess_type' must be either 'chat' or 
              'standard'
            ValueError: attempted to add a system role with history that
              already has it. To overwrite the existing role, set the
              'override_sys' flag to True
        """
        self.hist = copy.deepcopy(hist)
        self.system_role = system_role
        self.supports_sys = supports_sys

        if sess_type not in ['chat', 'chat_custom', 'standard']:
            raise ValueError(
                'argument \'sess_type\' must be either \'chat\' or '
                '\'chat_custom\' or \'standard\''
            )
        self.sess_type = sess_type

        # validate history.
        if len(self.hist) > 0:
            self._validate_hist()

        el = {'role': 'system', 'content': system_role}
        if system_role and len(self.hist) == 0:
            if self.supports_sys:
                self.hist.append(el)
            else:
                """
                system role gets added when the first 'add_prompt' is
                called
                """
                pass
        elif system_role and len(self.hist) > 0:
            if self.hist[0]['role'] != 'system':
                self.hist = [el] + self.hist
            elif override_sys:
                self.hist = [el] + self.hist[1:]
            else:
                raise ValueError(
                    "attempted to add a system role with history that "
                    "already has it. To overwrite the existing role, "
                    "set the 'override_sys' flag to True"
                )

    
    def __str__(self):
        return f'Session({self.hist})'

    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.hist)
    
    def _validate_hist(self):
        """validates chat history
        
        this validates the order of the chat history.
        role order should be:
        system (optional), user, assistant, user, assistant, etc.

        Raises:
            ValueError: history should not have more than a single turn 
              with the system role'
            ValueError: system role can only be in the first turn. role 
              was found at turn x
            ValueError: invalid chat order detected at turn: x
        """
        if len(self.hist) != 0:
            roles = [turn['role'] for turn in self.hist]
            has_system = False

            if 'system' in roles:
                has_system = True
                # check if system role appears more than once
                if roles.count('system') > 1:
                    raise ValueError(
                        'history should not have more than a single '
                        'turn with the system role'
                    )
                # check if system role appears in any other turn than 0
                elif roles.index('system') != 0:
                    raise ValueError(
                        f'system role can only be in the first turn. '
                        f'role was found at turn {roles.index("system")}'
                    )

                # system role passed checks so remove it from roles for
                # further validation
                roles = roles[1:]

            for turn, role in enumerate(roles):
                if (
                    turn%2 == 0 and role != 'user' 
                    or turn%2 == 1 and role != 'assistant'
                ):
                    raise ValueError(
                        f'invalid chat order detected at turn: '
                        f'{turn+has_system}'
                    )



    def get_hist(self) -> List[Dict[str, str]] | List[str]:
        """returns the chat history"""
        return self.hist
    
    def get_last(self) -> str | None:
        """Returns the last message in the session history

        Returns the last message in the session history. If there is
        nothing in the history, None is returned

        Raises:
            Exception: Unexpected case reached due to invalid Session.
        """
        if len(self.hist) == 0:
            return None
        elif self.sess_type in ['chat', 'chat_custom']:
            return self.hist[-1]['content']
        elif self.sess_type == 'standard':
            return self.hist[-1]
        else:
            raise Exception(
                'Unexpected case reached due to invalid Session'
            )
    def add_prompt(self, 
        prompt: str, 
        role: str='user',
        inplace: bool=False
    ) -> Optional["Session"]:
        """adds a prompt to the session

        takes a prompt and appends it to the end of the session history.
        'role' is only used if self.sess_type=='chat'.

        Example usage:
          >>> sess = Session('chat')
          >>> sess.add_prompt('hello')
          >>> print(sess.hist)
          [{'role': 'user', 'content': 'hello'}]
        """
        if (
            role == 'user' 
            and len(self.hist) == 0 
            and not self.supports_sys
        ):
            el = {
                'role': role, 
                'content': f"{self.system_role}\n\n{prompt}"
            }
        else:
            el = {'role': role, 'content': prompt}
        if inplace:
            self.hist.append(el)
            return self
        else:
            return Session(
                sess_type=self.sess_type, 
                hist=self.hist + [el]
            )
    
    def __add__(self, prompt: str):
        return self.add_prompt(prompt)

def create_sessions(
    N: int, 
    session_type: str
) -> List[Session]:
    """initializes new sessions

    returns a list of 'N' sessions depending on the 'session_type'

    Args:
        N: the number of sessions to be initialized
        session_type: the type of sessions to be initialized. Must 
            be one of ['chat', 'chat_custom', 'standard']
    Returns:
        A list of 'Session' objects.
        for example:

        [Session([]), Session([])]
    Raises:
        ValueError: 'session_type' must be one of ['chat', 'chat_custom'
          , 'standard']
    """

    if session_type not in ['chat', 'standard']:
        raise ValueError(
            '\'session_type\' must be one of [\'chat\', \'chat_custom\ '
            '\'standard\']'
        )
    raise NotImplementedError()

def apply_custom_format(
    format: Dict[str, str],
    sessions: Session | List[Session],
    for_reply: bool=True
) -> str | List[str]:
    r"""applies a custom format to a chat session or list of sessions

    takes in a dictionary of the format:
        {'system': <SYSTEM_FORMAT>,
         'user': <USER_FORMAT>,
         'assistant': <ASSISTANT_FORMAT>}
    and creates a string using the chat history of a session. adds a
    blank assistant token at the end to prepare for model generation.

    Example:
        >>> keys = {
        ...     'system': '[SYS]{{content}}[\SYS]\n'
        ...     'user': '[USER]{{content}}[\USER]\n'
        ...     'assistant': '[ASSISTANT]{{content}}[\ASSISTANT]\n'
        ... }
        >>> session = Session(hist=[
        ...     {'role': 'system', 
        ...      'content': 'you are a senior developer'},
        ...     {'role': 'user', 
        ...      'content': 'how do i print hello world in python'},
        ...     {'role': 'assistant', 
        ...      'content': 'print(\'hello world\')'},
        ... ])
        >>> text = apply_custom_format(keys, session)
        >>> print(text)
        [SYS] you are a senior developer [\SYS]
        [USER] how do i print hello world in python[\USER]
        [ASSISTANT] print('hello world') [\ASSISTANT]

    Args:
        format: a dictionary of formats that correspond to the possible
          roles. if no 'system' role is defined, it is used in the
          content of the first 'user' turn
        sessions: a singleton or list of Session objects. 
        for_reply: 'True' if you want to add a blank assistant token at
          the end of the formatted string. (Default: True)

    Returns:
        a singleton or list of strings that have applied the chat 
          template
    """
    if isinstance(sessions, list):
        return [
            _apply_format(format, session, for_reply)
            for session in sessions
        ]
    else:
        return _apply_format(format, sessions, for_reply)


def _apply_format(
        format: Dict[str, str], 
        session: Session, 
        for_reply: bool=True):
    """applies the chat template to a single session

    Applies the chat template to a single session. history valididty
    is not checked here because it checked at initialization or 
    when it is altered.

    Args:
        format: a dictionary of formats that correspond to the possible
          roles. if no 'system' role is defined, it is used in the
          content of the first 'user' turn
        session: a Session object
    
    Returns:
        a string that follows the provided chat template
    """
    
    out_text = ''
    carryover = ''
    hist = session.get_hist()
    roles = [turn['role'] for turn in hist]
    contents = [turn['content'] for turn in hist]
    if len(roles) == 0:
        return ''

    if for_reply and roles[-1] == 'assistant':
        raise ValueError(
            f'attempted to add the \'for_reply\' element but the last '
            f'turn in the chat already has an \'assistant\' role. '
            f'perpetrating session:\n{session}'
        )
        
    
    # if no system format is defined, roll system prompt over to first
    # user content
    if roles[0] == 'system' and 'system' not in format:
        roles = roles[1:]
        sys_content = contents.pop(0)
        contents[0] = f'{sys_content}\n\n{contents[0]}'

    for role, content in zip(roles, contents):
        out_text += strings.replace_slots(
            format[role], {'content': content}
        )
    
    if for_reply:
        out_text += format['for_reply']

    return out_text.strip()