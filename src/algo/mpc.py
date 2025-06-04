from scipy.optimize import minimize
import numpy as np


class ApcMPCAlgorithm:
    def __init__(
        self,
        ouput_hzn: list,
        input_hzn: list,
        knob_count: int,
        horizon_length: int,
        target: list,
        rate_matrix: list,
        state_r: list,
        state_q: list,
        uspec_lower: list,
        uspec_upper: list,
        udelta_lower: list,
        udelta_upper: list,
        control_q: list | None = None,
    ):
        """Model Predict Control Algorithm of APC.\n
        APC MPC is a robust and sohpisticated model predict control algorithm.\n
        Its State Space Model is as follows:\n
        x(k+1) = A*x(k) + B*u(k) + F*w(k)\n
        y(k) = C*x(k) + v(k)\n
        where:\n
        x(k) is the state vector at time k\n
        u(k) is the input vector at time k\n
        w(k) is the process noise vector at tim\n
        y(k) is the output vector at time k\n
        v(k) is the measurement noise vector at time\n
        Matrix A: [[0.0, 0.0], [0.0, 1.0]]\n
        Matrix B: [[rate], [0.0]]\n
        Matrix C: [[1.0, 0.0]\n
        Matrix F: [[1.0], [0.0]]\n

        Args:
            ouput_hzn (np.ndarray): _description_
            input_hzn (np.ndarray): _description_
        """
        self.output_hzn = ouput_hzn
        self.input_hzn = input_hzn
        self.knob_count = knob_count
        self.output_count_act = knob_count
        self.output_count_max = knob_count

        self.horizon_length = horizon_length
        self.target = target
        self.rate_matrix = rate_matrix
        assert len(self.rate_matrix) == self.knob_count * self.knob_count  # rate_matrix must be a square matrix

        self.state_r = state_r
        self.state_q = state_q
        self.uspec_lower = uspec_lower
        self.uspec_upper = uspec_upper
        self.udelta_lower = udelta_lower
        self.udelta_upper = udelta_upper

        if control_q is None:
            self.control_q = [1000.0 * self.knob_count]
        else:
            self.control_q = control_q

        self.generate_matrix_for_input()
        self.generate_matrix_abcf()

    def generate_matrix_for_input(self) -> None:
        self.matrix_rate = np.reshape(self.rate_matrix, (self.knob_count, self.knob_count))
        self.matrix_target = np.reshape(self.target * self.horizon_length, (self.knob_count * self.horizon_length, 1))

        self.matrix_output = np.reshape(self.output_hzn, (self.knob_count * self.horizon_length, 1))
        self.matrix_input = np.reshape(self.input_hzn, (self.knob_count * self.horizon_length, 1))

        self.matrix_state_r = np.diag(self.state_r * self.horizon_length)
        self.matrix_state_q = np.diag(self.state_q * self.horizon_length)

        self.matrix_control_q = np.diag(self.control_q * self.horizon_length)

        self.matrix_uspec_lower = np.reshape(
            self.uspec_lower * self.horizon_length, (self.knob_count * self.horizon_length, 1)
        )
        self.matrix_uspec_upper = np.reshape(
            self.uspec_upper * self.horizon_length, (self.knob_count * self.horizon_length, 1)
        )
        self.matrix_udelta_lower = np.reshape(self.udelta_lower, (self.knob_count, 1))
        self.matrix_udelta_upper = np.reshape(self.udelta_upper, (self.knob_count, 1))

    def generate_matrix_abcf(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        basic_eye_matrix = np.eye(self.knob_count)

        # Matrix A
        unit_a = np.zeros_like(basic_eye_matrix)
        upper_a = np.hstack((unit_a, unit_a))
        under_a = np.hstack((unit_a, basic_eye_matrix))
        self.matrix_a = np.vstack((upper_a, under_a))

        # Matrix B
        under_b = np.zeros_like(self.matrix_rate)
        self.matrix_b = np.vstack((self.matrix_rate, under_b))

        # Matrix C
        self.matrix_c = np.hstack((basic_eye_matrix, basic_eye_matrix))

        # Matrix F
        unit_f = np.zeros_like(basic_eye_matrix)
        self.matrix_f = np.vstack((unit_f, basic_eye_matrix))

        return self.matrix_a, self.matrix_b, self.matrix_c, self.matrix_f

    def generate_matrix_uwx(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """generate matrix u, w, x.\n
        matrix_u = [CA, 0, 0, 0, ..., 0]\n
        matrix_w = [CF, 0, 0, 0, ..., 0]\n
        matrix_x = [0, 0, 0, 0, ..., 0]\n

        Args:
            m (int): _description_
            n (int): _description_

        Returns:
            tuple: _description_
        """
        # zero matrix in Matrix U, W, X
        zero_matrix_row_count = self.matrix_c.shape[0]
        w_zero_matrix_col_count = self.matrix_f.shape[1]
        u_zero_matrix_col_count = self.matrix_b.shape[1]
        zero_matrix_u = np.zeros((zero_matrix_row_count, u_zero_matrix_col_count))
        zero_matrix_w = np.zeros((zero_matrix_row_count, w_zero_matrix_col_count))

        matrix_x_row_list = []
        matrix_u_row_list = []
        matrix_w_row_list = []

        for i in range(0, self.horizon_length):
            # Matrix_x row = [C*A^(i+1)]
            matrix_a_power = np.linalg.matrix_power(a=self.matrix_a, n=i + 1)
            matrix_x_row = self.matrix_c @ matrix_a_power
            matrix_x_row_list.append(matrix_x_row)

            # Matrix_u row = [C*A^(i)*B, C*A(i-1)*B, C*A^(i-2)*B,..., 0, 0, 0,..., 0]
            # for each element in a row
            matrix_u_single_row_element_list = []
            for j in range(i + 1):
                if i - j > 0:
                    matrix_a_power = np.linalg.matrix_power(a=self.matrix_a, n=i - j)
                    matrix_u_row_element = self.matrix_c @ matrix_a_power @ self.matrix_b
                    matrix_u_single_row_element_list.append(matrix_u_row_element)
                elif i - j == 0:
                    matrix_u_row_element = self.matrix_c @ self.matrix_b
                    matrix_u_single_row_element_list.append(matrix_u_row_element)
            # Behind N - i element are zero like matrix
            matrix_u_single_row_element_list += [zero_matrix_u for _ in range(self.horizon_length - i - 1)]

            matrix_u_single_row = np.hstack(matrix_u_single_row_element_list)
            matrix_u_row_list.append(matrix_u_single_row)

            # Matrix_w row = [C*A^(i)*F, C*A(i-1)*F, C*A^(i-2)*F,..., 0, 0, 0,..., 0]
            # for each element in a row
            matrix_w_single_row_element_list = []
            for j in range(i + 1):
                if i - j > 0:
                    matrix_a_power = np.linalg.matrix_power(a=self.matrix_a, n=i - j)
                    matrix_w_row_element = self.matrix_c @ matrix_a_power @ self.matrix_f
                    matrix_w_single_row_element_list.append(matrix_w_row_element)
                elif i - j == 0:
                    matrix_w_row_element = self.matrix_c @ self.matrix_f
                    matrix_w_single_row_element_list.append(matrix_w_row_element)
            # Behind N - i element are zero like matrix
            matrix_w_single_row_element_list += [zero_matrix_w for _ in range(self.horizon_length - i - 1)]

            matrix_w_single_row = np.hstack(matrix_w_single_row_element_list)
            matrix_w_row_list.append(matrix_w_single_row)

        self.matrix_u = np.vstack(matrix_u_row_list)
        self.matrix_w = np.vstack(matrix_w_row_list)
        self.matrix_x = np.vstack(matrix_x_row_list)

        return self.matrix_u, self.matrix_w, self.matrix_x

    def generate_initial_state_x0(self) -> np.ndarray:
        one_matrix_target = np.reshape(self.target, (self.knob_count, 1))
        input_0 = self.matrix_input[: self.knob_count]

        a = self.matrix_rate @ input_0
        b = one_matrix_target - a
        self.x_0 = np.vstack((a, b))

        return self.x_0

    def generate_noise_w(self) -> np.ndarray:
        lhs = self.matrix_state_r @ self.matrix_w.T @ self.matrix_w + self.matrix_state_q
        lhs_inv = np.linalg.inv(lhs)

        middle_b = self.matrix_state_r @ self.matrix_w.T
        middle_c = self.matrix_output - self.matrix_u @ self.matrix_input - self.matrix_x @ self.x_0
        rhs = middle_b @ middle_c

        # self.noise_w = np.linalg.solve(lhs, rhs)
        self.noise_w = lhs_inv @ rhs

        return self.noise_w

    def generate_recommend_setting(self) -> np.ndarray:
        lhs = np.sqrt(self.matrix_control_q) @ self.matrix_u
        lhs_pinv = np.linalg.pinv(lhs)
        rhs = self.matrix_x @ self.x_0 + self.matrix_w @ self.noise_w - self.matrix_target

        u_vector = -(lhs_pinv @ np.sqrt(self.matrix_control_q) @ rhs)

        return u_vector

    def auglag_optimize_with_constrain(self):
        tmpval = self.matrix_x @ self.x_0 + self.matrix_w @ self.noise_w - self.matrix_target
        input_0 = self.matrix_input[: self.knob_count]

        def quadratic_cost_function(input_hzn: np.ndarray) -> float:
            matrix = self.matrix_u @ input_hzn + tmpval
            matrix_sum = (np.sum(matrix)) ** 2
            return matrix_sum

        def gradient_function(input_hzn: np.ndarray) -> np.ndarray:
            matrix = self.matrix_u @ input_hzn + tmpval
            grad = self.matrix_u.T @ matrix
            gradient = 2 * grad
            return gradient

        def constrain_function(input_hzn: np.ndarray) -> np.ndarray:
            t1 = input_hzn - self.matrix_uspec_lower
            t2 = self.matrix_uspec_upper - input_hzn
            conditions: list = [t1, t2]

            first_delta = input_hzn[: self.knob_count] - input_0
            conditions += [(first_delta - self.matrix_udelta_lower), (self.matrix_udelta_upper - first_delta)]

            if self.horizon_length >= 2:
                for i in range(2, self.horizon_length + 1):
                    idx1 = (i - 2) * self.knob_count
                    idx2 = (i - 1) * self.knob_count + 1
                    delta_step_i = input_hzn[idx1 + self.knob_count : idx2 + self.knob_count] - input_hzn[idx1:idx2]
                    conditions.append(delta_step_i - self.matrix_udelta_lower)
                    conditions.append(self.matrix_udelta_upper - delta_step_i)

            flattened = np.concatenate(conditions).flatten()

            return flattened

        constraints = {"type": "ineq", "fun": constrain_function}
        result = minimize(
            fun=quadratic_cost_function,
            x0=self.matrix_input,
            jac=gradient_function,
            constraints=constraints,
            method="SLSQP",
            options={"disp": True},
        )
        return result

    def generate_cost_function(
        self,
    ):
        pass

    def optimization_cost_function(
        self,
    ):
        pass

    def optimization_with_constraints(
        self,
    ):
        pass

    def generate_prediction(
        self,
    ):
        pass
