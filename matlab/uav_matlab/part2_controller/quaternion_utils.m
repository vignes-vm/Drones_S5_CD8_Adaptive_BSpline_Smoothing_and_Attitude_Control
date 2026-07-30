classdef quaternion_utils
% All quaternion math in one place

    methods (Static)
        function q = euler2q(phi, theta, psi)
            q = [cos(phi/2)*cos(theta/2)*cos(psi/2)+sin(phi/2)*sin(theta/2)*sin(psi/2);
                 sin(phi/2)*cos(theta/2)*cos(psi/2)-cos(phi/2)*sin(theta/2)*sin(psi/2);
                 cos(phi/2)*sin(theta/2)*cos(psi/2)+sin(phi/2)*cos(theta/2)*sin(psi/2);
                 cos(phi/2)*cos(theta/2)*sin(psi/2)-sin(phi/2)*sin(theta/2)*cos(psi/2)];
        end

        function e = q2euler(q)
            q = q(:)/norm(q);
            e = [atan2(2*(q(1)*q(2)+q(3)*q(4)), q(1)^2-q(2)^2-q(3)^2+q(4)^2);
                 asin(max(-1,min(1, 2*(q(1)*q(3)-q(4)*q(2)))));
                 atan2(2*(q(1)*q(4)+q(2)*q(3)), q(1)^2+q(2)^2-q(3)^2-q(4)^2)];
        end

        function qm = mult(p, q)
            if length(p)==3, p=[0;p(:)]; end
            if length(q)==3, q=[0;q(:)]; end
            p=p(:); q=q(:);
            qm = [p(1)*q(1)-p(2)*q(2)-p(3)*q(3)-p(4)*q(4);
                  p(1)*q(2)+p(2)*q(1)+p(3)*q(4)-p(4)*q(3);
                  p(1)*q(3)-p(2)*q(4)+p(3)*q(1)+p(4)*q(2);
                  p(1)*q(4)+p(2)*q(3)-p(3)*q(2)+p(4)*q(1)];
        end

        function qc = conj(q)
            qc = [q(1);-q(2);-q(3);-q(4)];
        end

        function [dq, dw] = derivatives(q, w, tau, I)
            dq   = -0.5 * quaternion_utils.mult([0;w], q);
            gyro = cross(w, I*w);
            dw   = I \ (tau - gyro);
        end

        function [q_new, w_new] = rk4_step(q, w, tau, I, dt)
            [k1q,k1w] = quaternion_utils.derivatives(q, w, tau, I);
            [k2q,k2w] = quaternion_utils.derivatives(q+0.5*dt*k1q, w+0.5*dt*k1w, tau, I);
            [k3q,k3w] = quaternion_utils.derivatives(q+0.5*dt*k2q, w+0.5*dt*k2w, tau, I);
            [k4q,k4w] = quaternion_utils.derivatives(q+dt*k3q,     w+dt*k3w,     tau, I);
            q_new = q + (dt/6)*(k1q+2*k2q+2*k3q+k4q);
            w_new = w + (dt/6)*(k1w+2*k2w+2*k3w+k4w);
            q_new = q_new / norm(q_new);
        end

        function [tau, axis_err, err_mag] = p2_control(q_ref, q_meas, w, Pq, Pw, tau_max)
            q_err = quaternion_utils.mult(q_ref, quaternion_utils.conj(q_meas));
            if q_err(1) < 0, q_err = -q_err; end
            axis_err = q_err(2:4);
            tau = -Pq*axis_err - Pw*w;
            tau = max(-tau_max, min(tau_max, tau));
            err_mag = norm(axis_err);
        end
    end
end