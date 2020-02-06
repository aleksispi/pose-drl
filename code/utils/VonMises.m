classdef VonMises
    % von Mises distribution helper
    %
    % Functionalities: pdf, cdf, truncated pdf, sampling, truncated
    % sampling
    
    properties
        % No properties; "static class"
    end
    
    methods
        
        function obj = VonMises()
            % Empty constructor; "static class"
        end
        
        function [p, alpha] = circ_vmpdf(obj, alpha, thetahat, kappa)
            % Computes the von Mises pdf with preferred direction thetahat 
            % and concentration kappa at each of the angles in alpha
            %
            %   The pdf is given by f(phi) =
            %   (1 / (2pi * I0(kappa)) * exp(kappa * cos(phi - thetahat)
            %
            %   Input:
            %     alpha     angles to evaluate pdf at, if empty alphas are
            %               chosen to 100 uniformly spaced points around the
            %               circle
            %     [thetahat preferred direction, default is 0]
            %     [kappa    concentration parameter, default is 1]
            %
            %   Output:
            %     p         von Mises pdf evaluated at alpha
            %     alpha     angles at which pdf was evaluated
            %
            %
            %   References:
            %     Statistical analysis of circular data, Fisher
            %
            % Circular Statistics Toolbox for Matlab

            % By Philipp Berens and Marc J. Velasco, 2009
            % velasco@ccs.fau.edu

            % if no angles are supplied, 100 evenly spaced points around
            % the circle are chosen
            
            % Default args
            if ~exist('alpha', 'var') || isempty(alpha)
                alpha = linspace(0, 2 * pi, 101);
                alpha = alpha(1 : end - 1);
            end
            if ~exist('thetahat', 'var'); thetahat = 0; end
            if ~exist('kappa', 'var'); kappa = 1; end
            
            % Make alpha it is column vector
            alpha = alpha(:);

            % evaluate pdf
            p = 1/(2*pi*besseli(0, kappa)) * exp(kappa * cos(alpha - thetahat));
        end
        
        function cdf = circ_vmcdf(obj, x, a, b)
            % Evaluates the von Mises CDF.
            %
            %  Licensing:
            %
            %    This code is distributed under the GNU LGPL license.
            %
            %  Modified:
            %
            %    17 November 2006
            %
            %  Author:
            %
            %    Geoffrey Hill
            %
            %    MATLAB translation by John Burkardt.
            %
            %  Reference:
            %
            %    Geoffrey Hill,
            %    ACM TOMS Algorithm 518,
            %    Incomplete Bessel Function I0: The von Mises Distribution,
            %    ACM Transactions on Mathematical Software,
            %    Volume 3, Number 3, September 1977, pages 279-284.
            %
            %    Kanti Mardia, Peter Jupp,
            %    Directional Statistics,
            %    Wiley, 2000, QA276.M335
            %
            %  Parameters:
            %
            %    Input, real X, the argument of the CDF.
            %    A - PI <= X <= A + PI.
            %
            %    Input, real A, B, the parameters of the PDF.
            %    -PI <= A <= PI,
            %    0.0 < B.
            %
            %    Output, real CDF, the value of the CDF.
            %
            
            % Some numbers
            a1 = 12;
            a2 = 0.8;
            a3 = 8;
            a4 = 1;
            c1 = 56;
            ck = 10.5;
            
            % We expect -PI <= X - A <= PI
            if (x - a <= -pi); cdf = 0; return; end
            if (pi <= x - a);  cdf = 1; return; end
            
            % Convert angle (X - A) modulo 2 PI to the range (0, 2 * PI)
            z = b;
            u = mod(x - a + pi, 2 * pi);
            if (u < 0); u = u + 2 * pi; end

            y = u - pi;
            % For small B, sum IP terms by backwards recursion
            if (z <= ck)
                v = 0;
                if (0 < z)
                    ip = floor ( z * a2 - a3 / ( z + a4 ) + a1 );
                    p = ip;
                    s = sin ( y );
                    c = cos ( y );
                    y = p * y;
                    sn = sin ( y );
                    cn = cos ( y );
                    r = 0.0;
                    z = 2.0 / z;
                    for n = 2 : ip
                        p = p - 1.0;
                        y = sn;
                        sn = sn * c - cn * s;
                        cn = cn * c + y * s;
                        r = 1.0 / ( p * z + r );
                        v = ( sn / p + v ) * r;
                    end
                end
                cdf = ( u * 0.5 + v ) / pi;

            % For large B, compute the normal approximation and left tail
            else
                c = 24.0 * z;
                v = c - c1;
                r = sqrt ( ( 54.0 / ( 347.0 / v + 26.0 - c ) - 6.0 + c ) / 12.0 );
                z = sin ( 0.5 * y ) * r;
                s = 2.0 * z * z;
                v = v - s + 3.0;
                y = ( c - s - s - 16.0 ) / 3.0;
                y = ( ( s + 1.75 ) * s + 83.5 ) / v - y;
                arg = z * ( 1.0 - s / y^2 );
                erfx = obj.r8_error_f(arg);
                cdf = 0.5 * erfx + 0.5;
            end
            cdf = max(cdf, 0.0);
            cdf = min(cdf, 1.0);
            return
        end

        function value = r8_error_f(obj, x) %#ok<*INUSL>
            % Evaluates the error function ERF(X).
            %
            %  Discussion:
            %
            %    Since some compilers already supply a routine named ERF which evaluates
            %    the error function, this routine has been given a distinct, if
            %    somewhat unnatural, name.
            %    
            %    The function is defined by:
            %
            %      ERF(X) = ( 2 / sqrt ( PI ) ) * Integral ( 0 <= T <= X ) EXP ( - T^2 ) dT.
            %
            %    Properties of the function include:
            %
            %      Limit ( X -> -Infinity ) ERF(X) =          -1.0;
            %                               ERF(0) =           0.0;
            %                               ERF(0.476936...) = 0.5;
            %      Limit ( X -> +Infinity ) ERF(X) =          +1.0.
            %
            %      0.5 * ( ERF(X/sqrt(2)) + 1 ) = Normal_01_CDF(X)
            %
            %  Licensing:
            %
            %    This code is distributed under the GNU LGPL license.
            %
            %  Modified:
            %
            %    17 November 2006
            %
            %  Author:
            %
            %    Original FORTRAN77 version by William Cody.
            %    MATLAB version by John Burkardt.
            %
            %  Reference:
            %
            %    William Cody,
            %    "Rational Chebyshev Approximations for the Error Function",
            %    Mathematics of Computation,
            %    1969, pages 631-638.
            %
            %  Parameters:
            %
            %    Input, real X, the argument of the error function.
            %
            %    Output, real VALUE, the value of the error function.
            %
            a = [ ...
              3.16112374387056560E+00, ...
              1.13864154151050156E+02, ...
              3.77485237685302021E+02, ...
              3.20937758913846947E+03, ...
              1.85777706184603153E-01 ];
            b = [ ...
              2.36012909523441209E+01, ...
              2.44024637934444173E+02, ...
              1.28261652607737228E+03, ...
              2.84423683343917062E+03 ];
            c = [ ...
              5.64188496988670089E-01, ...
              8.88314979438837594E+00, ...
              6.61191906371416295E+01, ...
              2.98635138197400131E+02, ...
              8.81952221241769090E+02, ...
              1.71204761263407058E+03, ...
              2.05107837782607147E+03, ...
              1.23033935479799725E+03, ...
              2.15311535474403846E-08 ];
            d = [ ...
              1.57449261107098347E+01, ...
              1.17693950891312499E+02, ...
              5.37181101862009858E+02, ...
              1.62138957456669019E+03, ...
              3.29079923573345963E+03, ...
              4.36261909014324716E+03, ...
              3.43936767414372164E+03, ...
              1.23033935480374942E+03 ];
            p = [ ...
              3.05326634961232344E-01, ...
              3.60344899949804439E-01, ...
              1.25781726111229246E-01, ...
              1.60837851487422766E-02, ...
              6.58749161529837803E-04, ...
              1.63153871373020978E-02 ];
            q = [ ...
              2.56852019228982242E+00, ...
              1.87295284992346047E+00, ...
              5.27905102951428412E-01, ...
              6.05183413124413191E-02, ...
              2.33520497626869185E-03 ];
            sqrpi = 0.56418958354775628695;
            thresh = 0.46875;
            xbig = 26.543;
            xsmall = 1.11E-16;

            xabs = abs ( ( x ) );

            % Evaluate ERF(X) for |X| <= 0.46875.
            if ( xabs <= thresh )

                if ( xsmall < xabs )
                    xsq = xabs * xabs;
                else
                    xsq = 0.0;
                end

                xnum = a(5) * xsq;
                xden = xsq;
                for i = 1 : 3
                    xnum = ( xnum + a(i) ) * xsq;
                    xden = ( xden + b(i) ) * xsq;
                end

                erfxd = x * ( xnum + a(4) ) / (xden + b(4));
                
            %  Evaluate ERFC(X) for 0.46875 <= |X| <= 4.0.
            elseif ( xabs <= 4.0 )

                xnum = c(9) * xabs;
                xden = xabs;
                for i = 1 : 7
                    xnum = ( xnum + c(i) ) * xabs;
                    xden = ( xden + d(i) ) * xabs;
                end

                erfxd = ( xnum + c(8) ) / ( xden + d(8) );
                xsq = floor ( xabs * 16.0 ) / 16.0;
                del = ( xabs - xsq ) * ( xabs + xsq );
                erfxd = exp ( - xsq * xsq ) * exp ( - del ) * erfxd;

                erfxd = ( 0.5 - erfxd ) + 0.5;

                if ( x < 0.0 )
                    erfxd = - erfxd;
                end
                
            %  Evaluate ERFC(X) for 4.0 < |X|.
            else
                if ( xbig <= xabs )
                    if ( 0.0 < x )
                        erfxd = 1.0;
                    else
                        erfxd = -1.0;
                    end
                else
                    xsq = 1.0 / ( xabs * xabs );

                    xnum = p(6) * xsq;
                    xden = xsq;
                    for i = 1 : 4
                        xnum = ( xnum + p(i) ) * xsq;
                        xden = ( xden + q(i) ) * xsq;
                    end

                    erfxd = xsq * ( xnum + p(5) ) / ( xden + q(5) );
                    erfxd = ( sqrpi - erfxd ) / xabs;
                    xsq = floor ( xabs * 16.0 ) / 16.0;
                    del = ( xabs - xsq ) * ( xabs + xsq );
                    erfxd = exp ( - xsq * xsq ) * exp ( - del ) * erfxd;

                    erfxd = ( 0.5 - erfxd ) + 0.5;

                    if ( x < 0.0 )
                        erfxd = - erfxd;
                    end
                end
            end
            value = erfxd;
            return
        end
        
        function pdf = circ_vmpdf_trunc(obj, alpha, thetahat, kappa, trunc_interval)
            % Truncated von Mises pdf
            % Note: Assumes alpha is a SCALAR, not vector
            % Also note that if trunc_interval=[a,b] has a > b, then the
            % function assumes that the pdf should be truncated to something
            % like "-3pi/4 --> 3pi/4" (total pi/2 radians)
            % Finally, assumes max(trunc_interval) < pi and
            % min(trunc_interval) >= -pi

            % Defaults to regular pdf
            if ~exist('trunc_interval', 'var')
                pdf = circ_vmpdf(alpha, thetahat, kappa);
                return;
            end

            % Make sure alpha is in [-pi, pi) range
            alpha = angle(cos(alpha) + 1i * sin(alpha));

            % If outside truncated interval, pdf = 0
            if obj.is_outside(alpha, trunc_interval)
                pdf = 0;
                return;
            end

            % Finally we normalize the pdf within the trunc_interval appropriately
            a = trunc_interval(1); b = trunc_interval(2);
            if a <= b
                pdf = obj.circ_vmpdf(alpha, thetahat, kappa) / ...
                        (obj.circ_vmcdf(b, thetahat, kappa) - ...
                         obj.circ_vmcdf(a, thetahat, kappa));
            else
                pdf = obj.circ_vmpdf(alpha, thetahat, kappa) / ...
                        (obj.circ_vmcdf(abs(b) + 2 * abs(thetahat - abs(b)), ...
                                    thetahat, kappa) - ...
                         obj.circ_vmcdf(a, thetahat, kappa));    
            end
        end
        % Helper function to circ_vmpdf_trunc above
        function out = is_outside(obj, alpha, trunc_interval)
            a = trunc_interval(1); b = trunc_interval(2);
            if a <= b, out = alpha < a || alpha > b;
            else out = alpha > b && alpha < a;
            end
        end
       
        function alpha = circ_randvm(obj, theta, kappa, n, skip_intervals)

            % [p alpha] = circ_randvm(theta, kappa, n)
            %   Simulates n random angles from a von Mises distribution,
            %   with preferred direction theta and concentration parameter
            %   kappa.
            %
            %   Input:
            %     [theta           preferred direction, default is 0]
            %     [kappa           width, default is 1]
            %     [n               number of samples, default is 1]
            %     [skip_intervals: for truncated von Mises]
            %
            %   Output:
            %     alpha     samples from von Mises distribution
            %
            %   References:
            %     Statistical analysis of circular data, Fisher,
            %     sec. 3.3.6, p. 49
            %
            % Circular Statistics Toolbox for Matlab

            % By Marc J. Velasco, 2009
            % velasco@ccs.fau.edu
            % Distributed under Open Source BSD License
            %
            % Edited by Aleksis Pirinen (aleksis@maths.lth.se) 2018

            % Default args
            if ~exist('skip_intervals', 'var'); skip_intervals = []; end
            if ~exist('n', 'var'); n = 1; end
            if ~exist('kappa', 'var'); kappa = 1; end
            if ~exist('theta', 'var'); theta = 0; end

            % Approximate with uniform distribution if precision too low
            if kappa < 1e-6; alpha = 2 * pi * rand(n, 1); return; end

            % Other cases
            a = 1 + sqrt((1 + 4 * kappa.^2));
            b = (a - sqrt(2 * a)) / (2 * kappa);
            r = (1 + b^2) / (2 * b);
            alpha = zeros(n, 1);
            for j = 1 : n
                while 1
                    u = rand(3, 1);
                    z = cos(pi * u(1));
                    f = (1 + r * z) / (r + z);
                    c = kappa * (r - f);
                    if u(2) < c * (2-c) || u(2) <= c * exp(1-c); break; end
                end
                if u(3) > .5; alpha(j) = theta + acos(f);
                else alpha(j) = theta - acos(f); end
                alpha(j) = angle(exp(1i * alpha(j)));

                % Double check that not in skip-intervals (truncated von Mises)
                nbr_intervals = size(skip_intervals, 1);
                alpha_ok = 1;
                for i = 1 : nbr_intervals
                    interval = skip_intervals(i, :);
                    if interval(1) > interval(2)
                        if alpha(j) > interval(1) || alpha(j) < interval(2)
                            alpha_ok = 0;
                            break
                        end
                    else
                        if alpha(j) > interval(1) && alpha(j) < interval(2)
                            alpha_ok = 0;
                            break
                        end
                    end
                end
                while ~alpha_ok
                    while 1
                        u = rand(3, 1);
                        z = cos(pi * u(1));
                        f = (1 + r * z) / (r + z);
                        c = kappa * (r - f);
                        if u(2) < c * (2-c) || u(2) <= c * exp(1-c)
                            break;
                        end
                    end
                    if u(3) > .5; alpha(j) = theta + acos(f);
                    else alpha(j) = theta - acos(f); end
                    alpha(j) = angle(exp(1i * alpha(j)));

                    % Double check that not in skip-intervals (truncated von Mises)
                    nbr_intervals = size(skip_intervals, 1);
                    alpha_ok = 1;
                    for i = 1 : nbr_intervals
                        interval = skip_intervals(i, :);
                        if interval(1) > interval(2)
                            if alpha(j) > interval(1) || alpha(j) < interval(2)
                                alpha_ok = 0;
                                break
                            end
                        else
                            if alpha(j) > interval(1) && alpha(j) < interval(2)
                                alpha_ok = 0;
                                break
                            end
                        end
                    end
                end
            end
        end
    end
end