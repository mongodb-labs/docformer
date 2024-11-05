from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from mdbrtools.schema import Schema

from origami.utils.common import Symbol

INVALID = -1
NOOP = 0


class VPDA:
    """The VPDA (Vectorized Pushdown Automaton) class is an implementation of pushdown
    automata that operates on vectorized data. It provides efficient ways to process and
    analyze sequences of tokens which follow a context-free grammar. The VPDA class allows users
    to define and operate on pushdown automata with arbitrary states, transitions, and
    stack operations."""

    def __init__(
        self,
        input_alph: torch.LongTensor,
        symbol_alph: torch.LongTensor,
        state_alph: torch.LongTensor,
        pad_input: int = Symbol.PAD.value,
        start_symbol: int = Symbol.START.value,
        end_symbol: int = Symbol.END.value,
        start_state: int = Symbol.START.value,
        end_state: int = Symbol.END.value,
    ):
        self.device = "cpu"

        self.symbol_alph = symbol_alph
        self.input_alph = input_alph
        self.state_alph = state_alph

        # moving tensors to numpy
        if isinstance(self.symbol_alph, torch.Tensor):
            self.symbol_alph = self.symbol_alph.cpu().numpy()
        if isinstance(self.input_alph, torch.Tensor):
            self.input_alph = self.input_alph.cpu().numpy()
        if isinstance(self.state_alph, torch.Tensor):
            self.state_alph = self.state_alph.cpu().numpy()

        self.pad_input = pad_input
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.start_state = start_state
        self.end_state = end_state

        # for symbols, we prepend the NOOP = 0 position
        assert (symbol_alph > 0).all(), "Symbols must be greater than 0"
        self.symbol_alph = np.concatenate((np.zeros(1, dtype="long"), self.symbol_alph))

        # create inverse lookups
        self.inputs_to_idx = self._create_inverse_index(self.input_alph)
        self.symbols_to_idx = self._create_inverse_index(self.symbol_alph)
        self.states_to_idx = self._create_inverse_index(self.state_alph)

        # generate pop, push and state tensors  (state x symbol x input)
        self.pop_t = np.full(
            (len(state_alph), len(self.symbol_alph), len(self.input_alph)),
            INVALID,
            dtype="long",
        )
        self.push_t = np.full(
            (len(self.state_alph), len(self.symbol_alph), len(self.input_alph)),
            INVALID,
            dtype="long",
        )
        self.state_t = np.full(
            (len(self.state_alph), len(self.symbol_alph), len(self.input_alph)),
            INVALID,
            dtype="long",
        )

        self.n_stacks = None
        self.stacks = None
        self.states = None

        # insert default transition END -> END: PAD -> None, None
        self.insert_transition(self.end_state, self.end_state, self.pad_input)

    def _create_inverse_index(self, x: np.ndarray) -> np.ndarray:
        """Creates an inverse index on the input tensor. -1 indicates absence of input.

        The inverse index maps from values in the input tensor to their indices in the input.
        The returned tensor has the same number of elements as the maximum value in the input,
        and each index in the input maps to its corresponding index in the inverse index.

        Example:
          input = tensor([4, 1, 6, 2])
          output = tensor([-1, 1, 3, -1, 0, -1, 2])

        """

        if x.size == 0:
            return np.array([], dtype="long")
        rindex = np.ones(x.max() + 1, dtype="long") * INVALID
        rindex[x] = np.arange(x.shape[0], dtype="long")
        return rindex

    def top_of_stack(self, stacks: np.ndarray = None) -> np.ndarray:
        """
        Returns the top of the stack for each row in stacks, considering that negative
        values are interpreted as popping the symbol from the stack. If the stack for a given
        row is empty, it returns INVALID.

        Parameters:
            stacks (np.ndarray, optional): The stacks to use. Defaults to self.stacks.

        Returns:
            torch.Tensor: A tensor containing the top symbols on the stack.
        """

        if stacks is None:
            stacks = self.stacks

        top = np.full((stacks.shape[0],), INVALID)
        if stacks.size == 0:
            return top

        rcstacks = np.flip(stacks, axis=1).cumsum(axis=1)

        leftmost = np.where(
            rcstacks > 0,
            np.tile(np.arange(rcstacks.shape[1], dtype="float"), (rcstacks.shape[0], 1)),
            np.inf,
        )
        min_idx = leftmost.argmin(axis=1)
        min_val = leftmost.min(axis=1)
        valid_rows = min_val != np.inf
        top[valid_rows] = rcstacks[valid_rows, min_idx[valid_rows]]

        return top

    def insert_transition(
        self,
        source_state: int,
        target_state: int,
        input: int,
        pop_symbol: int = None,
        push_symbol: int = None,
    ) -> None:
        """
        Inserts a transition into the transition tables based on the given source state, target
        state, input, pop symbol, and push symbol.

        If no pop_symbol is given, this means the top of the stack is irrelevant and we insert
        a transition for all top_of_stack symbols for this transition.

        Parameters:
            source_state (int): The source state tensor.
            target_state (int): The target state tensor.
            input (int): The input tensor.
            pop_symbol (int, optional): The symbol to be popped. Defaults to None.
            push_symbol (int, optional): The symbol to be pushed. Defaults to None.

        Returns:
            None

        """

        # look up tensor positions in reverse indices
        source_id = self.states_to_idx[source_state]
        input_id = self.inputs_to_idx[input]

        if pop_symbol is None:
            top_of_stack_id = np.arange(len(self.symbol_alph), dtype="long")
        else:
            top_of_stack_id = self.symbols_to_idx[pop_symbol]

        # insert transitions
        self.pop_t[source_id, top_of_stack_id, input_id] = NOOP if pop_symbol is None else pop_symbol
        self.push_t[source_id, top_of_stack_id, input_id] = NOOP if push_symbol is None else push_symbol
        self.state_t[source_id, top_of_stack_id, input_id] = target_state

    def initialize(self, n_stacks: int) -> None:
        """
        Initialize the PDAs with the given number of stacks.

        Parameters:
            n_stacks (int): The number of stacks to initialize.

        Returns:
            None
        """
        self.n_stacks = n_stacks
        self.stacks = np.full((n_stacks, 1), self.start_symbol, dtype="long")
        self.states = np.full((n_stacks,), self.start_state, dtype="long")

    def get_input_mask(self) -> np.ndarray:
        """returns an array of shape (n_stacks, vocab_size) as a boolean mask of allowed next
        inputs, True indicates the input is allowed.

            Example:

            if the allowed indices are this:
                  [[True, False, True, False],
                   [False, True, False, True],
                   [True, True, False, False]]

            and self.inputs is [0, 5, 1, 3]

            This would return the following mask:

                [[True, True, False, False, False, False]
                 [False, False, False, True, False, True]
                 [True, False, False, False, False, True]]

            e.g. for the second row, the mask is true at positions 5 and 3.

        """

        # get current top of stacks
        top = self.top_of_stack()
        top_ids = self.symbols_to_idx[top]

        # convert states to state_ids
        state_ids = self.states_to_idx[self.states]

        # get allowed indices from pop_t (arbitrary, could be push_t or state_t)
        allowed_indices = self.pop_t[state_ids, top_ids] != INVALID

        # create output mask
        mask = np.zeros((self.n_stacks, self.inputs_to_idx.shape[0]), dtype="bool")

        # Use advanced indexing to convert the allowed indices back to input ID positions
        mask[
            np.expand_dims(np.arange(allowed_indices.shape[0]), axis=1),
            self.input_alph,
        ] = allowed_indices

        return mask

    def get_sequence_mask(self, inputs: np.ndarray) -> np.ndarray:
        """Returns a boolean mask of shape (n_stacks, n_steps, vocab_size)
        indicating which inputs are allowed at each step in the sequence."""

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()

        masks = []
        self.initialize(inputs.shape[0])

        try:
            for step in range(inputs.shape[1]):
                # get current input
                curr_input = inputs[:, step]

                # get allowed inputs for each row
                masks.append(self.get_input_mask())

                # update PDA with inputs
                self.next(curr_input)
        except AssertionError:
            print("inputs", inputs)
            print("decoded", self.encoder.decode(inputs))
            print("current input", curr_input)
            print("current input decoded", self.encoder.decode(curr_input))
            raise

        mask = np.stack(masks, axis=1)
        return mask

    def next(self, inputs: np.ndarray) -> np.ndarray:
        """transitions the PDAs into their next states given current inputs, and pops/pushes
        new symbols to the stack. Returns a boolean mask of shape (n_stacks,) indicating which
        rows have reached the end state.

            Parameters:
              inputs: 1D tensor of shape (n_stacks,)

            Returns:
              1D tensor of shape (n_stacks,) containing the current state of each stack.
              This is useful to check if all stacks have reached the end state.

        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()

        # get current top of stacks and convert to indices
        top = self.top_of_stack()
        top_ids = self.symbols_to_idx[top]

        # convert states and inputs to indices
        state_ids = self.states_to_idx[self.states]
        input_ids = self.inputs_to_idx[inputs]

        # look up pop/push and next states in the transition tensors
        pops = self.pop_t[state_ids, top_ids, input_ids]
        pushes = self.push_t[state_ids, top_ids, input_ids]
        states = self.state_t[state_ids, top_ids, input_ids]

        assert (pops != INVALID).all(), "Invalid transition found in pop_t."
        assert (pushes != INVALID).all(), "Invalid transition found in push_t."
        assert (states != INVALID).all(), "Invalid transition found in state_t."

        # update stacks (neg for pop, pos for push)
        self.stacks = np.concatenate((self.stacks, np.expand_dims(-pops, axis=1)), axis=1)
        self.stacks = np.concatenate((self.stacks, np.expand_dims(pushes, axis=1)), axis=1)

        # update state
        self.states = states

        res = self.states == self.end_state
        return res

    def accepts(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask of shape (n_stacks,) indicating which rows successfully
        reach the end state (are accepted by the PDA) and which end up in invalid states.

        Parameters:
            inputs: 2D tensor of shape (n_stacks, n_steps)

        Returns:
            valid: 1D tensor of shape (n_stacks,) indicating which rows are valid

        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()

        valid = np.ones(inputs.shape[0], dtype="bool")
        self.initialize(inputs.shape[0])

        for step in range(inputs.shape[1]):
            # get current input
            curr_input = inputs[:, step]

            # get allowed inputs for each row
            allowed_input = self.get_input_mask()

            # record invalid states
            invalid_rows = np.take_along_axis(allowed_input, np.expand_dims(curr_input, axis=1), axis=1) == False
            valid[invalid_rows.squeeze()] = False

            # transition invalid states to END state and remaining inputs to PAD
            curr_input[~valid] = self.pad_input
            self.states[~valid] = self.end_state

            # update PDA with inputs
            self.next(curr_input)

        return valid

    def display_transitions(
        self,
        input_labels: Optional[list[Any]] = None,
        symbol_labels: Optional[list[Any]] = None,
        state_labels: Optional[list[Any]] = None,
    ) -> None:
        """
        Display transition tables, optionally with specified labels for input, symbols, and states.
        If labels are not provided, default to using the alphabet integers.

        Parameters:
            input_labels (list): List of input labels
            symbol_labels (list): List of symbol labels
            state_labels (list): List of state labels

         Returns:
            None
        """

        pd.set_option("display.expand_frame_repr", False)

        # prepend 0 to symbol labels if given
        symbol_labels = self.symbol_alph.tolist() if symbol_labels is None else [0] + symbol_labels
        input_labels = input_labels or self.input_alph.tolist()
        state_labels = state_labels or self.state_alph.tolist()

        def lookup_symbols(x):
            if x == 0:
                return "ε"
            if x == -1:
                return "-"
            return symbol_labels[self.symbols_to_idx[x]]

        lookup_symbols = np.vectorize(lookup_symbols, otypes=["str"])
        lookup_states = np.vectorize(
            lambda x: "-" if x == -1 else state_labels[self.states_to_idx[x]],
            otypes=["str"],
        )

        # pop transition matrix
        for state in self.state_alph:
            sid = self.states_to_idx[state]
            state_label = state_labels[sid]
            print(f"in state {state_label}: rows = inputs, columns = top of stack \n")

            pop_t = lookup_symbols(self.pop_t[sid])[1:]
            df = pd.DataFrame(pop_t.T, columns=symbol_labels[1:], index=input_labels)
            print("pop transition matrix:\n", df, "\n")

            push_t = lookup_symbols(self.push_t[sid])[1:]
            df = pd.DataFrame(push_t.T, columns=symbol_labels[1:], index=input_labels)
            print("push transition matrix:\n", df, "\n")

            state_t = lookup_states(self.state_t[sid])[1:]
            df = pd.DataFrame(state_t.T, columns=symbol_labels[1:], index=input_labels)
            print("state transition matrix:\n", df, "\n")

            print("--------------------------------------------\n")

        pd.reset_option("display.expand_frame_repr")


from origami.preprocessing import StreamEncoder
from origami.utils.common import ArrayStart, FieldToken


class ObjectVPDA(VPDA):
    """VPDA for sequences of tokens for documents, as our DocTokenizerPipe produces.

    Requires access to the StreamEncoder to build the transition matrices.

    If a schema is provided, transition rules will ensure that only values seen for
    any given key are allowed for that key.

    """

    def __init__(
        self,
        encoder: StreamEncoder,
        schema: Optional[Schema] = None,
    ):
        self.encoder = encoder
        self.schema = schema

        self.field_token_ids = [i for ft, i in encoder.tokens_to_ids.items() if isinstance(ft, FieldToken)]
        # include UNKNOWN as possible field token
        self.field_token_ids.append(encoder.encode(Symbol.UNKNOWN.value))

        prim_value_token_ids = [
            i for pvt, i in encoder.tokens_to_ids.items() if not isinstance(pvt, (ArrayStart, FieldToken, Symbol))
        ]
        # UNKNOWN is treated as primitive value
        prim_value_token_ids.append(encoder.encode(Symbol.UNKNOWN.value))

        array_token_ids = set(i for at, i in encoder.tokens_to_ids.items() if isinstance(at, ArrayStart))

        if len(array_token_ids) > 0:
            max_array_length = max([at.size for at in encoder.tokens_to_ids.keys() if isinstance(at, ArrayStart)])

            # ArrayStart(0) is treated as primitive value
            array_0_token_id = encoder.encode(ArrayStart(0))
            prim_value_token_ids.append(array_0_token_id)

            if max_array_length >= 1:
                array_1_token_id = encoder.encode(ArrayStart(1))
            if max_array_length >= 2:
                array_2_token_id = encoder.encode(ArrayStart(2))

        else:
            max_array_length = 0

        # turn set into list
        array_token_ids = list(array_token_ids)

        # stack symbol alphabet consists of predefined symbols plus ArrayStarts + FieldTokens
        symbol_alph = np.array(
            [Symbol.START.value, Symbol.DOC.value, Symbol.END.value] + array_token_ids + self.field_token_ids
        )

        # input alphabet consists of all encoded inputs
        input_alph = np.array(list(encoder.tokens_to_ids.values()))

        # state alphabet consists of a fixed set of states
        state_alph = np.array(
            [
                Symbol.START.value,
                Symbol.END.value,
                Symbol.FIELD.value,
                Symbol.VALUE.value,
            ]
        )

        super().__init__(
            symbol_alph=symbol_alph,
            input_alph=input_alph,
            state_alph=state_alph,
            pad_input=Symbol.PAD.value,
            start_symbol=Symbol.START.value,
            end_symbol=Symbol.END.value,
            start_state=Symbol.START.value,
            end_state=Symbol.END.value,
        )

        # define transition rules for START state
        self.insert_transition(
            Symbol.START.value,
            Symbol.FIELD.value,
            input=Symbol.START.value,
            pop_symbol=Symbol.START.value,
            push_symbol=Symbol.DOC.value,
        )

        # allow pad at the beginning as no-op so we can left-pad sequences
        self.insert_transition(
            Symbol.START.value,
            Symbol.START.value,
            input=Symbol.PAD.value,
            pop_symbol=None,
            push_symbol=None,
        )

        # STATE -> END: read END, pop DOC, push END
        self.insert_transition(
            Symbol.FIELD.value,
            Symbol.END.value,
            input=Symbol.END.value,
            pop_symbol=Symbol.DOC.value,
            push_symbol=Symbol.END.value,
        )

        # FIELD -> VALUE: read FieldToken(n), pop -, push FieldToken(n)
        for ftid in self.field_token_ids:
            self.insert_transition(
                Symbol.FIELD.value,
                Symbol.VALUE.value,
                input=ftid,
                pop_symbol=None,
                push_symbol=ftid,
            )

            # define transition rules for VALUE state for each FieldToken
            if schema and ftid != Symbol.UNKNOWN.value:
                # if a schema is provided, only allow values seen for that field or UNKNOWN
                ft = encoder.decode(ftid)
                allowed_values = list(schema.get_prim_values(ft.name)) + [Symbol.UNKNOWN]
                value_token_ids = encoder.encode(allowed_values)
                if len(array_token_ids) > 0:
                    value_token_ids.append(array_0_token_id)
            else:
                value_token_ids = prim_value_token_ids

            # VALUE -> FIELD: read primitive value, pop FieldToken(n), push -
            for pvtid in value_token_ids:
                self.insert_transition(
                    Symbol.VALUE.value,
                    Symbol.FIELD.value,
                    input=pvtid,
                    pop_symbol=ftid,
                    push_symbol=None,
                )

        # define transition rules for arrays

        # read array 1, don't change the stack, stay on VALUE
        if max_array_length >= 1:
            self.insert_transition(
                Symbol.VALUE.value,
                Symbol.VALUE.value,
                input=array_1_token_id,
                pop_symbol=None,
                push_symbol=None,
            )

        # For array >= 3
        for i in range(3, max_array_length + 1):
            at = encoder.encode(ArrayStart(i))
            at_minus1 = encoder.encode(ArrayStart(i - 1))

            # read array(n>=3), pop -, push array(n>1)
            self.insert_transition(
                Symbol.VALUE.value,
                Symbol.VALUE.value,
                input=at,
                pop_symbol=None,
                push_symbol=at,
            )

            # read prim value, pop array(n), push array(n-1)
            self.insert_transition(
                Symbol.VALUE.value,
                Symbol.VALUE.value,
                input=np.array(prim_value_token_ids),
                pop_symbol=at,
                push_symbol=at_minus1,
            )

            # FIELD -> VALUE : read subdoc_end, pop array(n), push array(n-1)
            self.insert_transition(
                Symbol.FIELD.value,
                Symbol.VALUE.value,
                input=Symbol.SUBDOC_END.value,
                pop_symbol=at,
                push_symbol=at_minus1,
            )

        if max_array_length >= 2:
            # read array(n=2), pop -, push array(2)
            self.insert_transition(
                Symbol.VALUE.value,
                Symbol.VALUE.value,
                input=array_2_token_id,
                pop_symbol=None,
                push_symbol=array_2_token_id,
            )

            # read prim value, pop array(2), push -
            self.insert_transition(
                Symbol.VALUE.value,
                Symbol.VALUE.value,
                input=np.array(prim_value_token_ids),
                pop_symbol=array_2_token_id,
                push_symbol=None,
            )

            # FIELD -> VALUE : read subdoc_end, push -, pop array(2)
            self.insert_transition(
                Symbol.FIELD.value,
                Symbol.VALUE.value,
                input=Symbol.SUBDOC_END.value,
                pop_symbol=array_2_token_id,
                push_symbol=None,
            )

        # define transition rules for subdocuments

        # VALUE -> FIELD : read subdoc_start, push -, pop -
        self.insert_transition(
            Symbol.VALUE.value,
            Symbol.FIELD.value,
            input=Symbol.SUBDOC_START.value,
            pop_symbol=None,
            push_symbol=None,
        )

        # read subdoc_end, push -, pop FieldToken
        self.insert_transition(
            Symbol.FIELD.value,
            Symbol.FIELD.value,
            input=Symbol.SUBDOC_END.value,
            pop_symbol=np.array(self.field_token_ids),
            push_symbol=None,
        )

    def get_current_field_context(self) -> np.ndarray | torch.Tensor:
        """
        Returns the top most field symbol on the stack for each sample in the VALUE state,
        or INVALID for samples not in the VALUE state.
        """

        # clone self.stacks and replace all non-field tokens with 0
        stacked_field_tokens = np.copy(self.stacks)
        mask = np.isin(abs(self.stacks), self.field_token_ids)
        stacked_field_tokens[~mask] = 0

        top = self.top_of_stack(stacked_field_tokens)

        is_value_state = self.states == Symbol.VALUE.value
        top[~is_value_state] = INVALID

        return top

    def get_input_mask(self):
        # get current top of stacks
        top = self.top_of_stack()
        top_ids = self.symbols_to_idx[top]

        # Get the current top most field token on the stack for the fields that are in the VALUE state
        field_context = self.get_current_field_context()

        # swap out the top_ids with current field contexts
        # ignoring any where the state is not VALUE
        is_invalid = field_context == INVALID
        top_ids[~is_invalid] = self.symbols_to_idx[field_context[~is_invalid]]

        # convert states to state_ids
        state_ids = self.states_to_idx[self.states]

        # get allowed indices from pop_t (arbitrary, could be push_t or state_t)
        allowed_indices = self.pop_t[state_ids, top_ids] != INVALID

        # create output mask
        mask = np.zeros((self.n_stacks, self.inputs_to_idx.shape[0]), dtype="bool")

        # Use advanced indexing to convert the allowed indices back to input ID positions
        mask[np.expand_dims(np.arange(allowed_indices.shape[0]), axis=1), self.input_alph] = allowed_indices

        return mask

    def display_transitions(
        self,
    ) -> None:
        # get labels from encoder
        input_labels = self.encoder.decode(self.input_alph)
        symbol_labels = self.encoder.decode(self.symbol_alph[1:])
        state_labels = self.encoder.decode(self.state_alph)

        return super().display_transitions(input_labels, symbol_labels, state_labels)
